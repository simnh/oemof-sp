"""
Open Questions:

* What if no costs are associated with the first stage var? ->
vars are  not in first stage variables

"""
from oemof import solph
import mpisppy.utils.sputils as sputils
from mpisppy.opt.ef import ExtensiveForm

import os
import pandas as pd
import matplotlib.pyplot as plt


def _get_results(model):
    """
    Helper function to get the results from the solved model instances
    """
    _invest = {}
    results = solph.processing.convert_keys_to_strings(model.results())
    for i in ["wind", "gas", "pv"]:
        _invest[i] = results[(i, "electricity")]["scalars"]["invest"]
    return _invest


def build_model(scenario_data):
    """ """
    timeindex = pd.date_range("1/1/2022", periods=24, freq="H")

    energysystem = solph.EnergySystem(timeindex=timeindex)

    data = pd.read_csv(
        os.path.join(os.getcwd(), "energy", "data.csv"), index_col=0
    )[0 : len(timeindex)]

    bel = solph.buses.Bus(label="electricity")
    energysystem.add(bel)

    energysystem.add(
        solph.components.Sink(
            label="excess_bel", inputs={bel: solph.flows.Flow()}
        )
    )

    energysystem.add(
        solph.components.Source(
            label="wind",
            outputs={
                bel: solph.flows.Flow(
                    max=data["wind"].values,
                    investment=solph.Investment(ep_costs=120, stochastic=True),
                )
            },
        )
    )
    energysystem.add(
        solph.components.Source(
            label="pv",
            outputs={
                bel: solph.flows.Flow(
                    max=scenario_data,
                    investment=solph.Investment(ep_costs=5, stochastic=True),
                )
            },
        )
    )
    energysystem.add(
        solph.components.Source(
            label="gas",
            outputs={
                bel: solph.flows.Flow(
                    variable_costs=20,
                    investment=solph.Investment(ep_costs=60, stochastic=True),
                )
            },
        )
    )

    # create simple sink object representing the electrical demand
    energysystem.add(
        solph.components.Sink(
            label="demand",
            inputs={
                bel: solph.flows.Flow(
                    fix=data["demand_el"].values, nominal_value=1
                )
            },
        )
    )

    model = solph.StochasticModel(energysystem)

    return model


scenarios = pd.read_csv("energy/pv_scenarios.csv", index_col=0)
prob = pd.read_csv("energy/pv_prob.csv", index_col=0)

# solve models deterministic -------------------------------------------------
invest_deterministic = {}
for sc in scenarios:
    om = build_model(scenarios[sc].values)
    om.solve("cbc")
    invest_deterministic[sc] = get_results(om)
    print(om.objective())


def scenario_creator(scenario_name):
    """ """
    model = build_model(scenarios[scenario_name].values)
    sputils.attach_root_node(
        model, model.first_stage_objective_expression, model.first_stage_vars
    )
    model._mpisppy_probability = prob.at[scenario_name, "0"]
    return model


# solve extensive form --------------------------------------------------------
options = {"solver": "cbc", "nonant_for_fixed_vars": True}
ef = ExtensiveForm(options, scenarios.columns, scenario_creator)
results = ef.solve_extensive_form()
objval = ef.get_objective_value()
print("Extensive form obj value:", f"{objval:.1f}")

invest = {}
for sc in scenarios:
    invest[sc + "-sp"] = get_results(getattr(ef.ef, sc))

df = pd.concat(
    [pd.DataFrame(invest_deterministic), pd.DataFrame(invest)["sc0-sp"]],
    axis=1,
)
df.rename(columns={"sc0-sp": "SP"}, inplace=True)

ax = df.T.plot(kind="bar", color=["skyblue", "slategray", "gold"], grid=True)
ax.set_xlabel("Scenarios")
ax.set_ylabel("Installed capacitiy")


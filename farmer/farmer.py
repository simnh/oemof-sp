import mpisppy.utils.sputils as sputils
import pyomo.environ as pyo

import oemof.solph as solph
import pandas as pd
from oemof_visio import ESGraphRenderer

yields = {
    "good": dict(zip(["wheat", "corn", "beets"], [3, 3.6, 24])),
    "average": dict(zip(["wheat", "corn", "beets"], [2.5, 3, 20])),
    "bad": dict(zip(["wheat", "corn", "beets"], [2, 2.4, 16])),
}

planting_costs = {"wheat": 150, "corn": 230, "beets": 260}
purchase_costs = {"wheat": 238, "corn": 210}
revenue = {"wheat": 170, "corn": 150, "beets-fav": 36, "beets-unfav": 10}
minimum_keep = {"wheat": 200, "corn": 240, "beets": 0}
ub = {"beets-fav": 6000}


def build_solph_model(yields):
    """ """
    my_index = pd.date_range("1/1/2011", periods=1, freq="H")

    es = solph.EnergySystem(timeindex=my_index)

    buses = {
        "wheat": solph.buses.Bus(label="wheat"),
        "corn": solph.buses.Bus(label="corn"),
        "beets": solph.buses.Bus(label="beets"),
        "cornfield": solph.buses.Bus("cornfield", balanced=False),
        "wheatfield": solph.buses.Bus("wheatfield", balanced=False),
        "beetsfield": solph.buses.Bus("beetsfield", balanced=False),
    }
    es.add(*buses.values())

    for product in ["wheat", "corn", "beets"]:
        es.add(
            solph.components.Transformer(
                label=product + "_plant",
                inputs={
                    buses[product + "field"]: solph.flows.Flow(
                        variable_costs=planting_costs[product],
                        share=True,
                        firststage=True,
                    )
                },
                outputs={
                    buses[product]: solph.flows.Flow(
                        # variable_costs=0,
                        # firststage=True,
                    )
                },
                conversion_factors={buses[product]: yields[product]},
            )
        )
        es.add(
            solph.components.Sink(
                label=product + "_slack",
                inputs={
                    buses[product]: solph.flows.Flow(
                        min=minimum_keep[product], max=100000, nominal_value=1
                    )
                },
            )
        )
        # add buy, only for wheat and corn
        if product != "beets":
            es.add(
                solph.components.Source(
                    label=product + "_purchase",
                    outputs={
                        buses[product]: solph.flows.Flow(
                            variable_costs=purchase_costs[product]
                        )
                    },
                )
            )
            es.add(
                solph.components.Sink(
                    label=product + "_sale",
                    inputs={
                        buses[product]: solph.flows.Flow(
                            variable_costs=-revenue[product]
                        )
                    },
                )
            )
        else:
            es.add(
                solph.components.Sink(
                    label=product + "-fav_sale",
                    inputs={
                        buses[product]: solph.flows.Flow(
                            variable_costs=-revenue[product + "-fav"],
                            nominal_value=1,
                            max=ub.get(product + "-fav"),
                        )
                    },
                )
            )
            es.add(
                solph.components.Sink(
                    label=product + "-unfav_sale",
                    inputs={
                        buses[product]: solph.flows.Flow(
                            variable_costs=-revenue[product + "-unfav"]
                        )
                    },
                )
            )

    om = solph.StochasticModel(es)

    # limit total planting of wheat and corn
    om = solph.constraints.generic_integral_limit(
        om, keyword="share", limit=500
    )

    # other way to get the flows which share the limit ...
    # flows={(s,t):f for (s,t),f in om.es.flows().items()
    # if s.label in ["corn_plant", "wheat_plant"]}

    # expression generated for lhs of constraint
    # om.integral_limit_share.pprint()
    return om


# deterministic model for average scenario
om = build_solph_model(yields["average"])
om.write("model.lp", io_options={"symbolic_solver_labels": True})
gr = ESGraphRenderer(
    energy_system=om.es, filepath="energy_system", img_format="png"
)
gr.view()

om.solve(solver="cbc")
print("Deterministic obj value for average:", om.objective())

# stochastic versions --------------------------------------------------------
def scenario_creator(scenario_name):
    if scenario_name in ["good", "average", "bad"]:
        pass
    else:
        raise ValueError("Unrecognized scenario name")

    model = build_solph_model(yields[scenario_name])
    sputils.attach_root_node(
        model, model.first_stage_objective_expression, model.first_stage_vars
    )
    model._mpisppy_probability = 1.0 / 3

    return model


# extensive form --------------------------------------------------------

from mpisppy.opt.ef import ExtensiveForm
from mpisppy.utils import sputils

options = {"solver": "cbc", "nonant_for_fixed_vars": True}
all_scenario_names = ["good", "average", "bad"]
ef = ExtensiveForm(options, all_scenario_names, scenario_creator)
results = ef.solve_extensive_form()
objval = ef.get_objective_value()
print("Extensive form obj value:", f"{objval:.1f}")

# solve with progressive hedging --------------------------------------------
from mpisppy.opt.ph import PH

options = {
    "solvername": "cbc",
    "PHIterLimit": 5,
    "defaultPHrho": 10,
    "convthresh": 1e-7,
    "verbose": False,
    "display_progress": False,
    "display_timing": False,
    "iter0_solver_options": dict(),
    "iterk_solver_options": dict(),
}
all_scenario_names = ["good", "average", "bad"]
ph = PH(
    options,
    all_scenario_names,
    scenario_creator,
)
ph.ph_main()
variables = ph.gather_var_values_to_rank0()
for (scenario_name, variable_name) in variables:
    variable_value = variables[scenario_name, variable_name]
    print(scenario_name, variable_name, variable_value)

# if scenarios converge, we should get the results with the oemof
# standard way
solved_model_good_scenario = ph.local_scenarios["good"]
print(type(solved_model_good_scenario))
results = solved_model_good_scenario.results()

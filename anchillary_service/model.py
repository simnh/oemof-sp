"""


"""
from oemof import solph
import mpisppy.utils.sputils as sputils
from mpisppy.opt.ef import ExtensiveForm

import os
import pandas as pd
import matplotlib.pyplot as plt

import pyomo.environ as po


def _get_results(model):
    """
    Helper function to get the results from the solved model instances
    """
    _invest = {}
    results = solph.processing.convert_keys_to_strings(model.results())
    for i in ["wind", "gas", "storage"]:
        _invest[i] = results[(i, "electricity")]["scalars"]["invest"]
    return _invest


def build_model(ts_data, costs, txi):
    """ """
    timeindex = pd.date_range("1/1/2022", periods=5, freq="H")

    es = solph.EnergySystem(timeindex=timeindex)

    bel = solph.buses.Bus(label="electricity")
    es.add(bel)

    es.add(
        solph.components.Source(
            label="grid_buy",
            outputs={
                bel: solph.flows.Flow(
                    variable_costs=costs["operation"]["grid_buy"]
                )
            },
        )
    )
    es.add(
        solph.components.Sink(
            label="grid_sell",
            inputs={
                bel: solph.flows.Flow(
                    variable_costs=costs["operation"]["grid_sell"]
                )
            },
        )
    )

    es.add(
        solph.components.Source(
            label="wind",
            outputs={
                bel: solph.flows.Flow(
                    fix=ts_data["wind"].values,
                    investment=solph.Investment(
                        ep_costs=costs["investment"]["wind"], firststage=True
                    ),
                )
            },
        )
    )
    es.add(
        solph.components.Source(
            label="pv",
            outputs={
                bel: solph.flows.Flow(
                    fix=ts_data["pv"].values,
                    investment=solph.Investment(
                        ep_costs=costs["investment"]["pv"], firststage=True
                    ),
                )
            },
        )
    )

    es.add(
        solph.components.Sink(
            label="demand",
            inputs={
                bel: solph.flows.Flow(
                    fix=ts_data["demand"].values, nominal_value=1
                )
            },
        )
    )

    es.add(
        solph.components.GenericStorage(
            label="storage",
            inputs={
                bel: solph.flows.Flow(
                    variable_costs=costs["operation"]["storage_in"],
                    investment=solph.Investment(firststage=False),
                )
            },
            outputs={
                bel: solph.flows.Flow(
                    investment=solph.Investment(firststage=False)
                )
            },
            loss_rate=0.00,
            initial_storage_level=0,
            invest_relation_input_capacity=1 / 6,
            invest_relation_output_capacity=1 / 6,
            inflow_conversion_factor=1,
            outflow_conversion_factor=0.8,
            investment=solph.Investment(
                ep_costs=costs["investment"]["storage_energy"], firststage=True
            ),
        )
    )

    m = solph.StochasticModel(es)

    m.storage_in2 = po.Var(m.TIMESTEPS, within=po.NonNegativeReals)
    m.storage_out2 = po.Var(m.TIMESTEPS, within=po.NonNegativeReals)
    m.grid_sell2 = po.Var(m.TIMESTEPS, within=po.NonNegativeReals)
    m.grid_buy2 = po.Var(m.TIMESTEPS, within=po.NonNegativeReals)

    m.equate_stage_variables = po.ConstraintList()

    for t in range(txi):
        # until t_xi grid connection and storage use can't anticipate the request and have to match those that would be without any request
        m.equate_stage_variables.add(
            m.grid_buy2[t]
            == m.flow[m.es.groups["grid_buy"], m.es.groups["electricity"], t]
        )
        m.equate_stage_variables.add(
            m.grid_sell2[t]
            == m.flow[m.es.groups["electricity"], m.es.groups["grid_sell"], t]
        )
        m.equate_stage_variables.add(
            m.storage_in2[t]
            == m.flow[m.es.groups["electricity"], m.es.groups["storage"], t]
        )
        m.equate_stage_variables.add(
            m.storage_out2[t]
            == m.flow[m.es.groups["storage"], m.es.groups["electricity"], t]
        )

    # remove energy balance to create a new one based on txi...
    m.del_component("BusBlock")

    group = [m.es.groups["electricity"]]
    ins = {}
    outs = {}
    for n in group:
        ins[n] = [i for i in n.inputs]
        outs[n] = [o for o in n.outputs]

    def _busbalance_rule(model, txi=txi):
        for t in m.TIMESTEPS:
            if t <= txi:
                for g in group:
                    lhs = sum(m.flow[i, g, t] for i in ins[g])
                    rhs = sum(m.flow[g, o, t] for o in outs[g])
                    expr = lhs == rhs
                    # no inflows no outflows yield: 0 == 0 which is True
                    if expr is not True:
                        model.balance.add((g, t), expr)
            if t == txi:
                for g in group:
                    lhs = (
                        sum(
                            m.flow[i, g, t]
                            for i in ins[g]
                            if not "storage" in i.label
                        )
                        + m.storage_out2[t]
                    )
                    rhs = (
                        sum(
                            m.flow[g, o, t]
                            for o in outs[g]
                            if not "storage" in o.label
                        )
                        + m.storage_in2[t]
                    )
                    expr = lhs == rhs
                    # no inflows no outflows yield: 0 == 0 which is True
                    if expr is not True:
                        model.balance.add((g, t), expr)
            if t > txi:
                for g in group:
                    lhs = (
                        sum(
                            m.flow[i, g, t]
                            for i in ins[g]
                            if not i.label in ["grid_buy", "storage"]
                        )
                        + m.grid_buy2[t]
                        + m.storage_out2[t]
                    )
                    rhs = (
                        sum(
                            m.flow[g, o, t]
                            for o in outs[g]
                            if not o.label in ["grid_sell", "storage"]
                        )
                        + m.grid_sell2[t]
                        + m.storage_in2[t]
                    )
                    expr = lhs == rhs
                    # no inflows no outflows yield: 0 == 0 which is True
                    if expr is not True:
                        model.balance.add((g, t), expr)

    m.balance = po.Constraint(group, m.TIMESTEPS, noruleinit=True)
    m.balance_build = po.BuildAction(rule=_busbalance_rule)

    return m


df = pd.read_csv("anchillary_service/data.csv", index_col=0)
costs = {
    "investment": {"pv": 2, "wind": 10, "storage_energy": 100},
    "operation": {"storage_in": 0.001, "grid_sell": 3, "grid_buy": 5},
}


# solve models deterministic -------------------------------------------------
m = build_model(ts_data=df, costs=costs, txi=3)


m.balance.pprint()

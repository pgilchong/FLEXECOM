"""
    Types

Core data structures for the FLEXECOM energy community optimization model.

This module defines the immutable configuration types used throughout the model:
- `Scenario`: Optimization scenario parameters
- `BESS_specs`: Battery energy storage system specifications
- `EV_specs`: Electric vehicle specifications
- `ASHP_specs`: Air-source heat pump specifications
- `Financial_params`: Financial analysis parameters
"""
module Types

export Scenario, BESS_specs, EV_specs, ASHP_specs, Financial_params

"""
    Scenario

Defines an optimization scenario with technology deployment and pricing parameters.

# Fields
- `id::Int`: Unique scenario identifier
- `name::String`: Human-readable scenario name (auto-generated from parameters)
- `COEF::Int`: Allocation coefficient type (1=static, 2=variable by season/day, 3=dynamic hourly)
- `PV::Float64`: Nominal PV power per household (kWp)
- `GRID::Float64`: Grid price multiplier (1.0 = baseline prices)
- `BESS_price::Float64`: BESS investment cost multiplier
- `BESS_cap::Float64`: Total BESS capacity for the community (kWh)
- `EVnum::Int`: Number of electric vehicles in the community
- `ASHP::Float64`: ASHP nominal power per household (kW), 0 = disabled
- `DHW::Int`: Domestic hot water system inclusion (0 = disabled, 1 = enabled)
"""
struct Scenario
    id::Int
    name::String
    COEF::Int                # Coefficient type (1=static, 2=variable, 3=dynamic)
    PV::Float64              # Nominal PV power (kWp)
    GRID::Float64            # Grid price multiplier
    BESS_price::Float64      # BESS price multiplier
    BESS_cap::Float64        # BESS capacity (kWh)
    EVnum::Int               # Number of electric vehicles
    ASHP::Float64            # ASHP power (kW)
    DHW::Int                 # DHW inclusion (0/1)
end

"""
    BESS_specs

Battery Energy Storage System technical specifications.

# Fields
- `cap::Float64`: Unit capacity (kWh)
- `capAh::Float64`: Unit capacity in Amp-hours
- `DoD::Float64`: Depth of discharge (fraction, e.g., 0.95)
- `Eff::Float64`: Round-trip efficiency (fraction, e.g., 0.9)
- `V::Float64`: Nominal voltage (V)
- `Pmax::Float64`: Maximum charge/discharge power (kW)
- `Inv::Float64`: Unit investment cost (€)
- `Cycles::Int`: Expected lifetime in charge/discharge cycles
"""
struct BESS_specs
    cap::Float64             # Capacity (kWh)
    capAh::Float64           # Capacity in Ah
    DoD::Float64             # Depth of discharge
    Eff::Float64             # Efficiency
    V::Float64               # Voltage (V)
    Pmax::Float64            # Maximum power (kW)
    Inv::Float64             # Inverter cost (€)
    Cycles::Int              # Life cycles
end

"""
    EV_specs

Electric Vehicle technical and economic specifications.

# Fields
- `cap::Float64`: Battery capacity (kWh)
- `cons_km::Float64`: Energy consumption per kilometer (kWh/km)
- `Eff::Float64`: Charging efficiency (fraction)
- `Pch::Float64`: Maximum charging power (kW)
- `cons_fuel::Float64`: Equivalent ICEV fuel consumption (L/km) for baseline comparison
- `ICEV_price::Float64`: Reference ICEV purchase price (€) for comparison
- `RP_price::Float64`: Recharge point installation cost (€)
"""
struct EV_specs
    cap::Float64             # Capacity (kWh)
    cons_km::Float64         # Consumption (kWh/km)
    Eff::Float64             # Efficiency
    Pch::Float64             # Charging power (kW)
    cons_fuel::Float64       # ICEV fuel consumption (L/km)
    ICEV_price::Float64      # ICEV base price (€)
    RP_price::Float64        # Recharge point price (€)
end

"""
    ASHP_specs

Air-Source Heat Pump technical specifications.

# Fields
- `Eff::Float64`: Nominal efficiency (COP at reference conditions)
- `price::Float64`: Unit investment cost (€)
"""
struct ASHP_specs
    Eff::Float64             # Nominal efficiency
    price::Float64           # ASHP price (€)
end

"""
    Financial_params

Parameters for techno-economic analysis of the energy community.

# Fields
- `d::Float64`: Discount rate for NPV calculations (e.g., 0.02 = 2%)
- `N::Int`: Analysis horizon in years
- `PV_CAPEX::Float64`: PV system capital cost (€/kWp)
- `INV_life::Int`: Inverter expected lifetime (years)
- `fuel_price::Float64`: Natural gas reference price (€/kWh)
- `OM_PV::Float64`: PV O&M cost (€/kWp/year)
- `OM_BESS::Float64`: BESS O&M as fraction of investment per year
- `OM_RP::Float64`: Recharge point O&M as fraction of investment per year
- `OM_ASHP::Float64`: ASHP O&M as fraction of investment per year
- `OM_DHW::Float64`: DHW system O&M as fraction of investment per year
- `DHW_price::Float64`: DHW tank investment cost (€)
- `PI_power::Float64`: Grid power term price (€/kW/year)
"""
struct Financial_params
    d::Float64               # Discount rate
    N::Int                   # Time horizon (years)
    PV_CAPEX::Float64        # PV cost (€/kWp)
    INV_life::Int            # Inverter lifetime (years)
    fuel_price::Float64      # Natural gas price (€/kWh)
    OM_PV::Float64           # O&M PV (% of investment/year)
    OM_BESS::Float64         # O&M BESS (% of investment/year)
    OM_RP::Float64           # O&M recharge point (% of investment/year)
    OM_ASHP::Float64         # O&M ASHP (% of investment/year)
    OM_DHW::Float64          # O&M DHW (% of investment/year)
    DHW_price::Float64       # DHW tank price (€)
    PI_power::Float64        # Power term price (€/kW/year)
end

end # module
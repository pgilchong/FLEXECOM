module Constants

using ..Types: BESS_specs, EV_specs, ASHP_specs, Financial_params

export FINANCIAL_DEFAULT,
       GRID_TERM_POWER,
       HOURS_PER_YEAR,
       DEFAULT_BESS,
       BESS_MIN_SOC,
       DEFAULT_EV,
       EV_SOC_MAX,
       DEFAULT_ASHP,
       COP_HEATING_COEFFS,
       COP_COOLING_COEFFS,
       cop_from_temp,
       DHW_COP,
       DHW_MAX_LITERS,
       DHW_SUPPLY_TEMPERATURE_K,
       DHW_INLET_TEMPERATURE_K,
       WATER_SPECIFIC_HEAT_KJKG,
       DHW_ENERGY_PER_LITER,
       DEFAULT_GUROBI_PARAMS,
       MONTHLY_HOUR_RANGES,
       INVERTER_COST_SLOPE,
       INVERTER_COST_INTERCEPT,
       RECHARGE_POINT_COST,
       ASHP_BASE_PRICE,
       PRICE_SENSITIVITY,
       PRICE_SENSITIVITY_KEYS,
       PRICE_SENSITIVITY_MEANS,
       DEFAULT_SAMPLE_DAYS

const FINANCIAL_DEFAULT = Financial_params(
    0.02,       # Discount rate
    20,         # Time horizon (years)
    650.0,      # PV cost (€/kWp)
    10,         # Inverter lifetime (years)
    0.02,       # Natural gas price (€/kWh) - default, scenarios may override fuel price
    20.0,       # O&M PV (€/kWp/year)
    0.02,       # O&M BESS (% of investment/year)
    0.02,       # O&M recharge point (% of investment/year)
    0.02,       # O&M ASHP (% of investment/year)
    0.01,       # O&M DHW (% of investment/year)
    131.0,      # DHW tank price (€)
    32.097019   # Power term price (€/kW/year)
)

const GRID_TERM_POWER = FINANCIAL_DEFAULT.PI_power
const HOURS_PER_YEAR = 8760

const DEFAULT_BESS = BESS_specs(3.552, 74, 0.95, 0.9, 48, 3.552, 1245, 6000)
const BESS_MIN_SOC = 0.2

const DEFAULT_EV = EV_specs(42, 0.159, 0.90, 7.4, 0.064, 0.0, 843.60)
const EV_SOC_MAX = 0.8

const DEFAULT_ASHP = ASHP_specs(1.0, 890.0)

const COP_HEATING_COEFFS = (a = 0.002610, b = 0.1618, c = 3.698)
const COP_COOLING_COEFFS = (a = 0.016, b = -1.384, c = 32.26)

@inline function cop_from_temp(coeffs, temp_celsius)
    coeffs.a * temp_celsius^2 + coeffs.b * temp_celsius + coeffs.c
end

const DHW_COP = 1.0
const DHW_MAX_LITERS = 80.0
const DHW_SUPPLY_TEMPERATURE_K = 55 + 273
const DHW_INLET_TEMPERATURE_K = 15 + 273
const WATER_SPECIFIC_HEAT_KJKG = 4.18
const DHW_ENERGY_PER_LITER = WATER_SPECIFIC_HEAT_KJKG *
    (DHW_SUPPLY_TEMPERATURE_K - DHW_INLET_TEMPERATURE_K) / 3600

const DEFAULT_GUROBI_PARAMS = Dict(
    "OutputFlag" => 1,
    "Method" => 2,
    "BarConvTol" => 1e-6,
    "Threads" => 6,
    "BarHomogeneous" => 1,
)

const MONTHLY_HOUR_RANGES = (
    1:744,
    745:1416,
    1417:2160,
    2161:2880,
    2881:3624,
    3625:4344,
    4345:5088,
    5089:5832,
    5833:6552,
    6553:7296,
    7297:8016,
    8017:8760,
)

const INVERTER_COST_SLOPE = 68.657
const INVERTER_COST_INTERCEPT = 668.5
const RECHARGE_POINT_COST = 843.60
const ASHP_BASE_PRICE = 890.0

const DEFAULT_SAMPLE_DAYS = [1, 90, 180, 270]

const PRICE_SENSITIVITY = (
    PV_price = (min = 0.40, mean = 0.80, max = 1.25, count = 9),
    Fuel_price = (min = 1.20, mean = 1.80, max = 2.40, count = 9),
    NG_price = (min = 0.05, mean = 0.08, max = 0.11, count = 9),
    EV_price = (min = 0.0, mean = 5000.0, max = 10000.0, count = 9),
    ASHP_price = (min = 0.50, mean = 0.75, max = 1.00, count = 9),  # MATLAB default: 0.75
)

const PRICE_SENSITIVITY_KEYS = (:PV_price, :Fuel_price, :NG_price, :EV_price, :ASHP_price)

const PRICE_SENSITIVITY_MEANS = (
    PV_price = PRICE_SENSITIVITY.PV_price.mean,
    Fuel_price = PRICE_SENSITIVITY.Fuel_price.mean,
    NG_price = PRICE_SENSITIVITY.NG_price.mean,
    EV_price = PRICE_SENSITIVITY.EV_price.mean,
    ASHP_price = PRICE_SENSITIVITY.ASHP_price.mean,
)

end # module

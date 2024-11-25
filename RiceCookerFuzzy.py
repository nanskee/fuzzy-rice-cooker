import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Step 1: Define fuzzy variables and membership functions

# Input variables
water_level = ctrl.Antecedent(np.arange(0, 11, 1), 'water_level')  # Water level (0-10 cups)
rice_quantity = ctrl.Antecedent(np.arange(0, 11, 1), 'rice_quantity')  # Rice quantity (0-10 cups)
rice_type = ctrl.Antecedent(np.arange(1, 4, 1), 'rice_type')  # Rice type (1=short, 2=long, 3=sticky)

# Output variable
cooking_time = ctrl.Consequent(np.arange(0, 61, 1), 'cooking_time')  # Cooking time in minutes (0-60 minutes)

# Membership functions for water level
water_level['low'] = fuzz.trimf(water_level.universe, [0, 0, 5])
water_level['medium'] = fuzz.trimf(water_level.universe, [0, 5, 10])
water_level['high'] = fuzz.trimf(water_level.universe, [5, 10, 10])

# Membership functions for rice quantity
rice_quantity['low'] = fuzz.trimf(rice_quantity.universe, [0, 0, 5])
rice_quantity['medium'] = fuzz.trimf(rice_quantity.universe, [0, 5, 10])
rice_quantity['high'] = fuzz.trimf(rice_quantity.universe, [5, 10, 10])

# Membership functions for rice type (1=short, 2=long, 3=sticky)
rice_type['short'] = fuzz.trimf(rice_type.universe, [1, 1, 1])
rice_type['long'] = fuzz.trimf(rice_type.universe, [2, 2, 2])
rice_type['sticky'] = fuzz.trimf(rice_type.universe, [3, 3, 3])

# Membership functions for cooking time
cooking_time['short'] = fuzz.trimf(cooking_time.universe, [0, 0, 30])
cooking_time['medium'] = fuzz.trimf(cooking_time.universe, [20, 30, 40])
cooking_time['long'] = fuzz.trimf(cooking_time.universe, [30, 60, 60])

# Step 2: Define fuzzy rules

rule1 = ctrl.Rule(water_level['low'] & rice_quantity['low'] & rice_type['short'], cooking_time['short'])
rule2 = ctrl.Rule(water_level['low'] & rice_quantity['medium'] & rice_type['short'], cooking_time['medium'])
rule3 = ctrl.Rule(water_level['low'] & rice_quantity['high'] & rice_type['short'], cooking_time['long'])

rule4 = ctrl.Rule(water_level['medium'] & rice_quantity['low'] & rice_type['long'], cooking_time['short'])
rule5 = ctrl.Rule(water_level['medium'] & rice_quantity['medium'] & rice_type['long'], cooking_time['medium'])
rule6 = ctrl.Rule(water_level['medium'] & rice_quantity['high'] & rice_type['long'], cooking_time['long'])

rule7 = ctrl.Rule(water_level['high'] & rice_quantity['low'] & rice_type['sticky'], cooking_time['medium'])
rule8 = ctrl.Rule(water_level['high'] & rice_quantity['medium'] & rice_type['sticky'], cooking_time['long'])
rule9 = ctrl.Rule(water_level['high'] & rice_quantity['high'] & rice_type['sticky'], cooking_time['long'])

# Step 3: Create the control system and simulation

cooking_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
cooking_sim = ctrl.ControlSystemSimulation(cooking_ctrl)

# Input values for the simulation
water_input = 6   # Example: 6 cups of water
rice_input = 4    # Example: 4 cups of rice
rice_type_input = 3  # Example: sticky rice (3)

cooking_sim.input['water_level'] = water_input
cooking_sim.input['rice_quantity'] = rice_input
cooking_sim.input['rice_type'] = rice_type_input

# Perform the calculation
cooking_sim.compute()

# Get the output result
print(f'Cooking Time: {cooking_sim.output["cooking_time"]:.2f} minutes')

# Step 4: Visualize the membership functions and results

# Plot membership functions for water level
water_level.view()

# Plot membership functions for rice quantity
rice_quantity.view()

# Plot membership functions for rice type
rice_type.view()

# Plot membership functions for cooking time
cooking_time.view()

# Plot the result
cooking_time.view(sim=cooking_sim)
plt.show()

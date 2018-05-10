# The following looks like its okay of assuming 'None or Unspecified' means
# either data missing or not applicable
unordered_cats = [
    'datasource',
    'auctioneerID',
    'fiBaseModel',
    'fiSecondaryDesc',
    'fiModelSeries'
    'fiModelDescriptor',
    'state',
    'ProductGroup',
    'ProductGroupDesc',
    'Drive_System',
    'Enclosure',
    'Forks',
    'Pad_Type',
    'Ride_Control',
    'Stick',
    'Transmission',
    'Turbocharged',
    'Blade_Extension',
    'Enclosure_Type',
    'Engine_Horsepower',
    'Hydraulics',
    'Pushblock',
    'Ripper',
    'Scarifier',
    'Tip_Control',
    'Coupler',
    'Coupler_System',
    'Grouser_Tracks',
    'Track_Type',
    'Thumb',
    'Pattern_Changer',
    'Grouser_Type',
    'Backhoe_Mounting',
    'Blade_Type',
    'Travel_Controls',
    'Differential_Type',
    'Steering_Controls',
]

# (Discrete) Categories with a natural order and more than 2 elements
ordered_cats = {
    'YearMade': int,
    'UsageBand': ['Low', 'High', 'Medium'],
    # If we assume that Mini vs Compact vs Small is not used for the same type of
    # products, their order should not matter
    'ProductSize': ['Mini', 'Compact', 'Small', 'Medium', 'Large / Medium', 'Large'],
    'Blade_Width': int,
    'Tire_Size': float,
    'Hydraulics_Flow': ['No', 'Standard'
                        'High Flow'],
    'Undercarriage_Pad_Width': float,
    'Stick_Length': float,
    'Grouser_Type': ['Single', 'Double', 'Triple']
}

continuous_vars = ['saledate', 'MachineHoursCurrentMeter']

# Has entries for 'Yes' but none for 'No'.
# Assume that 'None or Unspecified' means 'No' instead.
# Can also try the assumption that missing data means no instead.
assume_no = [
    'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow', 'Thumb',
    'Backhoe_Mounting', 'Turbocharged', 'Blade_Extension', 'Pushblock', 'Scarifier',
    'Forks'
]

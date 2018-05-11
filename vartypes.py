# The following looks like its okay of assuming 'None or Unspecified' means
# either data missing or not applicable
unordered_cats = [
    'datasource',
    'auctioneerID',
    'fiBaseModel',
    'fiSecondaryDesc',
    'fiModelSeries',
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

# (A, Bn) Treat none_unspec as own cat instead of missing data. NaN is treated as missing.
# If the var has two cats excluding none_unspec, create a ordered cat with
# none_unspec in the middle
none_as_ocat = [
    'Backhoe_Mounting', 'Blade_Extension', 'Coupler', 'Coupler_System', 'Enclosure_Type',
    'Forks', 'Grouser_Tracks', 'Hydraulics_Flow', 'Pattern_Changer', 'Pushblock',
    'Ride_Control', 'Scarifier', 'Thumb', 'Tip_Control', 'Turbocharged'
]

# (Bu) Cats without non_unspec
nan_as_cat = [
    'Engine_Horsepower',
    'Stick',
    'Track_Type',
]

# (C) Categories with a natural order and more than 2 elements (not counting none_unspec)
# none_unspec and NaN treaded as missing
ordered_multi_cats = {
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

# (E)
continuous_vars = ['saledate', 'MachineHoursCurrentMeter']

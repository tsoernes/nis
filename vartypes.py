# Create cat for none_unspec
none_unspec_cats = [
    'Backhoe_Mounting',
    'Blade_Extension',
    'Blade_Type',
    'Coupler',
    'Coupler_System',
    'Enclosure',
    'Enclosure_Type',
    'Forks',
    'Grouser_Tracks',
    'Hydraulics',
    'Pad_Type',
    'Pattern_Changer',
    'Pushblock',
    'Ride_Control',
    'Ripper',
    'Scarifier',
    'Thumb',
    'Tip_Control',
    'Transmission',
    'Travel_Controls',
    'Turbocharged',
]

# Cats without none_unspec
no_none_unspec_cats = [
    'fiSecondaryDesc',
    'fiModelSeries',
    'fiModelDescriptor',
    'fiBaseModel',
    'datasource',
    'auctioneerID',
    'Differential_Type',
    'Steering_Controls',
    'ProductGroupDesc',
    'ProductGroup',
    'Drive_System',
    'Engine_Horsepower',
    'state',
    'Stick',
    'Track_Type',
]

# Cats which are binary COUNTING potential none_unspec
bin_cats = [
    'Backhoe_Mounting', 'Blade_Extension', 'Coupler_System', 'Forks', 'Grouser_Tracks',
    'Pushblock', 'Scarifier', 'Turbocharged', 'Engine_Horsepower', 'Stick', 'Track_Type'
]

unordered_multi_cats = set(none_unspec_cats).union(
    set(no_none_unspec_cats)) - set(bin_cats)

# Categories with a natural order and
# more than 2 elements (not counting none_unspec or NaN)
# none_unspec and NaN treated as missing
ordered_multi_cats = {
    'saledate': int,
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

continuous_vars = ['MachineHoursCurrentMeter']

# None of the above groups should intersect
# svars = [
#     set(unordered_multi_cats),
#     set(no_none_unspec_cats),
#     set(ordered_multi_cats.keys()),
# ]
# for i, svar1 in enumerate(svars):
#     for j, svar2 in enumerate(svars[i + 1:]):
#         intsec = svar1.intersection(svar2)
#         assert len(intsec) == 0, (intsec, i, j)

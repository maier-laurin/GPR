# Spritpreise.csv
prices for gasoline at a lot of gasstations in austria.
x & y are coordinates in meters in the Austrian Labert projection, but recentered so that the origin lies in 5550 Radstadt, a town that lies rougthly in the middle of austria.
p are the prices in Euro at the gasstation to the corresponding coordinates.
the data got scraped from https://www.spritpreisrechner.at/ a webside provided by the econtrol.
the scraping took place in the evening of the 14th of November 2025.
requests where made for on a on roughtly a 7 km grid over austria, the website returnes the closest 10 gasstations to the requested point,
so except in very densly populated region all the gasstations which had their gasoline prices reported to the E-Controll on said evening should be included in the dataset.

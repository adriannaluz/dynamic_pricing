import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import levene
from scipy.stats import f_oneway

pd.set_option("display.expand_frame_repr", False)

data = pd.read_csv("data/dynamic_pricing.csv")
print(data.columns)
print(data.describe().T)

cat = []
num = []

for i, type in enumerate(data.dtypes):
    if type == "int64" or type == "float64":
        num.append(data.columns[i])
    else:
        cat.append(data.columns[i])

print(num)
# === Exploratory Data Analysis === #
# sns.pairplot(data)
# plt.savefig("pairplot_alldata.png")
# plt.close()

# sns.scatterplot(data,
#                 x='Expected_Ride_Duration',
#                 y='Historical_Cost_of_Ride',
#                 hue='Vehicle_Type')
# plt.savefig("rideDuration_rideCost_carType.png")
# plt.close()

cat_data = data.loc[:, cat]
num_data = data.loc[:, num]

# for category in cat_data.columns:
#     sns.barplot(data, x=category, y="Historical_Cost_of_Ride")
#     plt.savefig(f"rideCost_{category}.png")

premium_cost = data["Historical_Cost_of_Ride"].iloc[
    data.groupby("Vehicle_Type")._get_index("Premium")
]
economy_cost = data["Historical_Cost_of_Ride"].iloc[
    data.groupby("Vehicle_Type")._get_index("Economy")
]

# sns.histplot(premium_cost)
# sns.histplot(economy_cost)
# plt.savefig("rideCost_premium_economy_cars.png")

# === t-test === #
t_test = stats.ttest_ind(economy_cost, premium_cost)
print(t_test)  # T

silver_cost = data["Historical_Cost_of_Ride"].iloc[
    data.groupby("Customer_Loyalty_Status")._get_index("Silver")
]
regular_cost = data["Historical_Cost_of_Ride"].iloc[
    data.groupby("Customer_Loyalty_Status")._get_index("Regular")
]
gold_cost = data["Historical_Cost_of_Ride"].iloc[
    data.groupby("Customer_Loyalty_Status")._get_index("Gold")
]

# sns.histplot(silver_cost)
# sns.histplot(regular_cost)
# sns.histplot(gold_cost)
# plt.savefig("rideCost_silver_regular_gold_costumers.png")

# === Levene's test to test homogeneity of variance === #
lev_test = levene(
    silver_cost, regular_cost, gold_cost, center="mean", proportiontocut=0.05
)
print(lev_test)

anova = f_oneway(silver_cost, regular_cost, gold_cost)
print(anova)

# Correlation matrix
corr_matrix = num_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_matrix, cmap="viridis", vmax=0.8, square=True)
plt.savefig("correlation_matrix.png")

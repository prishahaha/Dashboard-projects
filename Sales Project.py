#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[41]:


sheets = pd.read_excel("E:\My Documents\Downloads\Regional Sales Dataset.xlsx",sheet_name=None)
df_sales       = sheets['Sales Orders']
df_customers   = sheets['Customers']
df_products    = sheets['Products']
df_regions     = sheets['Regions']
df_state_reg   = sheets['State Regions']
df_budgets     = sheets['2017 Budgets']


# In[7]:


print(f"df_sales      shape:{df_sales.shape}")
print(f"df_customers  shape: {df_customers.shape}")
print(f"df_products   shape: {df_products.shape} ")
print(f"df_regions    shape: {df_regions.shape} ")
print(f"df_state_reg  shape: {df_state_reg.shape} ")
print(f"df_budgets    shape: {df_budgets.shape}")


# In[42]:


df_sales.head()


# In[9]:


df_customers.head()


# In[43]:


df_regions.head()


# In[57]:


df_state_reg.head()


# In[56]:


new_header=df_state_reg.iloc[0] #grab 1st row
df_state_reg.columns=new_header #set columns
df_state_reg=df_state_reg[1:].reset_index(drop=True) #drop header row and reset


# In[45]:


df_products.head()


# In[46]:


df_budgets.head()


# In[47]:


df_sales.isnull().sum()


# In[48]:


len(df_sales)==len(df_sales.drop_duplicates())


# In[58]:


# Merge with customers
df=df_sales.merge(
   df_customers,
   how='left',
   left_on='Customer Name Index',
   right_on='Customer Index'
)
# Merge with products
df=df.merge(
   df_products,
   how='left',
   left_on='Product Description Index',
   right_on='Index'
)
# Merge with regions
df=df.merge(
   df_regions,
   how='left',
   left_on='Delivery Region Index',
   right_on='id'
)
# Merge with state regions
df=df.merge(
   df_state_reg[["State Code","Region"]],
   how='left',
   left_on='state_code',
   right_on='State Code'
)
# Merge with budgets
df=df.merge(
   df_budgets,
   how='left',
   on='Product Name',
)
df.head(1)


# In[28]:


df.to_excel('file.xlsx', index=False)


# In[59]:


cols_to_drop=['Unnamed: 12','Unnamed: 13', 'Customer Index','Index','id','State Code','type','population','households','time_zone','land_area','water_area','median_income']
df=df.drop(columns=cols_to_drop,errors='ignore')
df.head(1)


# In[60]:


df.columns=df.columns.str.lower()
df.columns.values


# In[61]:


cols_to_keep = [
    'ordernumber',        # unique order ID
    'orderdate',          # date when the order was placed
    'customer names',     # customer who placed the order
    'channel',            # sales channel (e.g., Wholesale, Distributor)
    'product name',       # product purchased
    'order quantity',     # number of units ordered
    'unit price',         # price per unit
    'line total',         # revenue for this line item (qty × unit_price)
    'total unit cost',    # company’s cost for this line item
    'state_code',         # two-letter state code
    'state',              # full state name
    'region',             # broader U.S. region (e.g., South, West)
    'latitude',           # latitude of delivery city
    'longitude',          # longitude of delivery city
    '2017 budgets'        # budget target for this product in 2017
]
df=df[cols_to_keep]
df = df.rename(columns={
    'ordernumber'      : 'order_number',   # snake_case for consistency
    'orderdate'        : 'order_date',     # date of the order
    'customer names'   : 'customer_name',  # customer who placed it
    'product name'     : 'product_name',   # product sold
    'order quantity'   : 'quantity',       # units sold
    'unit price'       : 'unit_price',     # price per unit in USD
    'line total'       : 'revenue',        # revenue for the line item
    'total unit cost'  : 'cost',           # cost for the line item
    'state_code'       : 'state',          # two-letter state code
    'state'            : 'state_name',     # full state name
    'region'           : 'us_region',      # broader U.S. region
    'latitude'         : 'lat',            # latitude (float)
    'longitude'        : 'lon',            # longitude (float)
    '2017 budgets'     : 'budget'          # 2017 budget target (float)
})
df.head()


# In[62]:


df.loc[df['order_date'].dt.year!=2017,'budget']=pd.NA
df[['order_date','product_name','revenue','budget']].head(10)


# In[63]:


df.info()


# In[64]:


df.describe()


# In[65]:


df.isnull().sum()


# In[69]:


df['total_cost']=df['quantity']*df['cost']
df['profit']=df['revenue']-df['total_cost']
df['profit_margin_pct']=(df['profit']/df['revenue'])*100
df['order_month_name'] = df['order_date'].dt.month_name()
df['order_month_num'] = df['order_date'].dt.month
df


# In[74]:


df['order_month'] = df['order_date'].dt.to_period('M')
monthly_sales=df.groupby('order_month')['revenue'].sum()
plt.figure(figsize=(15,4))

# Plot the monthly sales trend with circle markers and navy line
monthly_sales.plot(marker='o', color='navy')

# Scale y-axis values to millions for readability
from matplotlib.ticker import FuncFormatter
formatter = FuncFormatter(lambda x, pos: f'{x/1e6:.1f}M')
plt.gca().yaxis.set_major_formatter(formatter)

# Add title and axis labels
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Revenue (Millions)')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent clipping
plt.tight_layout()
plt.show()


# In[75]:


df_ = df[df['order_date'].dt.year != 2018] 
# 2. Group by month number and month name, sum revenue, then sort by month number
monthly_sales = (
    df_
    .groupby(['order_month_num', 'order_month_name'])['revenue']
    .sum()
    .sort_index()
)

# 3. Plot setup
from matplotlib.ticker import FuncFormatter

plt.figure(figsize=(13, 4))
plt.plot(
    monthly_sales.index.get_level_values(1),  # X-axis: month names
    monthly_sales.values,                     # Y-axis: total revenue
    marker='o',                                # circle markers
    color='navy'                               # line color
)

# 4. Scale y-axis values to millions for readability
formatter = FuncFormatter(lambda x, pos: f'{x/1e6:.1f}M')
plt.gca().yaxis.set_major_formatter(formatter)

# 5. Add title and axis labels
plt.title('Overall Monthly Sales Trend (Excluding 2018)')
plt.xlabel('Month')
plt.ylabel('Total Revenue (Millions)')

# 6. Rotate x-axis labels for readability
plt.xticks(rotation=45)

# 7. Adjust layout to prevent clipping
plt.tight_layout()

# 8. Display the plot
plt.show()


# In[76]:


# Calculate total revenue for each product and convert values to millions
top_prod = df.groupby('product_name')['revenue'].sum() / 1_000_000

# Select the top 10 products by revenue
top_prod = top_prod.nlargest(10)

# Set the figure size for clarity
plt.figure(figsize=(9, 4))

# Plot a horizontal bar chart: x-axis as revenue in millions, y-axis as product names
sns.barplot(
    x=top_prod.values,    # X-axis: revenue values in millions
    y=top_prod.index,     # Y-axis: product names
    palette='viridis'     # Color palette for bars
)

# Add title and axis labels
plt.title('Top 10 Products by Revenue (in Millions)')  # Main title
plt.xlabel('Total Revenue (in Millions)')              # X-axis label
plt.ylabel('Product Name')                             # Y-axis label

# Adjust layout to prevent overlapping elements
plt.tight_layout()

# Display the plot
plt.show()


# In[77]:


# Calculate total revenue for each product and convert values to millions
top_prod = df.groupby('product_name')['revenue'].sum() / 1_000_000

# Select the top 10 products by revenue
top_prod = top_prod.nsmallest(10)

# Set the figure size for clarity
plt.figure(figsize=(9, 4))

# Plot a horizontal bar chart: x-axis as revenue in millions, y-axis as product names
sns.barplot(
    x=top_prod.values,    # X-axis: revenue values in millions
    y=top_prod.index,     # Y-axis: product names
    palette='magma'     # Color palette for bars
)

# Add title and axis labels
plt.title('Bottom 10 Products by Revenue (in Millions)')  # Main title
plt.xlabel('Total Revenue (in Millions)')              # X-axis label
plt.ylabel('Product Name')                             # Y-axis label

# Adjust layout to prevent overlapping elements
plt.tight_layout()

# Display the plot
plt.show()


# In[78]:


top_margin = (
    df.groupby('product_name')['profit']
      .mean()                        # Calculate mean profit for each product
      .sort_values(ascending=False)  # Sort from highest to lowest average profit
      .head(10)                      # Keep only the top 10 products
)
plt.figure(figsize=(9, 4))

# Plot a horizontal bar chart: x-axis as revenue in millions, y-axis as product names
sns.barplot(
    x=top_margin.values,    # X-axis: revenue values in millions
    y=top_margin.index,     # Y-axis: product names
    palette='mako'     # Color palette for bars
)

# Add title and axis labels
plt.title('Top 10 Products by Avg Profit Margin')  # Main title
plt.xlabel('Avg Profit Margin (in USD)')              # X-axis label
plt.ylabel('Product Name')                             # Y-axis label

# Adjust layout to prevent overlapping elements
plt.tight_layout()

# Display the plot
plt.show()


# In[79]:


chan_sales = df.groupby('channel')['revenue'].sum().sort_values(ascending=False)

# Set figure size for the pie chart
plt.figure(figsize=(5, 5))

# Plot pie chart with percentage labels and a defined start angle
plt.pie(
    chan_sales.values,                   # Data: revenue values per channel
    labels=chan_sales.index,             # Labels: channel names
    autopct='%1.1f%%',                   # Display percentages with one decimal
    startangle=140,                      # Rotate chart so first slice starts at 140 degrees
    colors=sns.color_palette('coolwarm') # Color palette for slices
)

# Add title for context
plt.title('Total Sales by Channel')

# Adjust layout to ensure everything fits well
plt.tight_layout()

# Display the chart
plt.show()


# In[84]:


# Calculate the total revenue for each order to get the order value
aov = df.groupby('order_number')['revenue'].sum()

# Set the figure size for better visibility
plt.figure(figsize=(12, 4))

# Plot a histogram of order values
plt.hist(
    aov,               # Data: list of order values
    bins=50,           # Number of bins to group order values
    color='plum',   # Fill color of the bars
    edgecolor='black'  # Outline color of the bars
)

# Add title and axis labels for context
plt.title('Distribution of Average Order Value')
plt.xlabel('Order Value (USD)')
plt.ylabel('Number of Orders')

# Adjust layout to prevent clipping
plt.tight_layout()

# Show the plot
plt.show()
     


# In[86]:


plt.figure(figsize=(6,4))

# Plot unit price vs. profit margin percentage
plt.scatter(
    df['unit_price'],            # X-axis: unit price in USD
    df['profit_margin_pct'],     # Y-axis: profit margin percentage
    alpha=0.6,                   # Transparency level for overlapping points
    color='lightseagreen'                # Point color
)

# Add title and axis labels
plt.title('Profit Margin % vs. Unit Price')  # Chart title
plt.xlabel('Unit Price (USD)')                # X-axis label
plt.ylabel('Profit Margin (%)')               # Y-axis label

# Adjust layout to prevent clipping
plt.tight_layout()

# Display the plot
plt.show()


# In[87]:


# Set figure size for clarity
plt.figure(figsize=(12,4))

# Create a boxplot of unit_price by product_name
sns.boxplot(
    data=df,
    x='product_name',   # X-axis: product categories
    y='unit_price',      # Y-axis: unit price values
    color='c'            # Box color
)

# Add title and axis labels
plt.title('Unit Price Distribution per Product')  # Chart title
plt.xlabel('Product')                              # X-axis label
plt.ylabel('Unit Price (USD)')                     # Y-axis label

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Display the plot
plt.show()


# In[88]:


top_rev=(df.groupby('customer_name')['revenue'].sum().sort_values(ascending=False).head(10))
bottom_rev=(df.groupby('customer_name')['revenue'].sum().sort_values(ascending=True).head(10))
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.barplot(
    x=top_rev.values/1e6,
    y=top_rev.index,
    palette='Blues_r',
    ax=axes[0])
axes[0].set_title('Top 10 customers by Revenue',fontsize=14)
axes[0].set_xlabel('Revenue (in millions)',fontsize=12)
axes[0].set_ylabel('Customer Name',fontsize=12)
sns.barplot(
    x=bottom_rev.values/1e6,
    y=bottom_rev.index,
    palette='Reds',
    ax=axes[1])
axes[1].set_title('Bottom 10 customers by Revenue',fontsize=14)
axes[1].set_xlabel('Revenue (in millions)',fontsize=12)
axes[1].set_ylabel('Customer Name',fontsize=12)
plt.tight_layout()
plt.show()


# In[93]:


apc=(df.groupby('channel')['profit_margin_pct'].mean().sort_values(ascending=False))
plt.figure(figsize=(6,4))
ax=sns.barplot(
       x=apc.index,
       y=apc.values,
       palette='coolwarm')
plt.title('Average Profit Margin by Channel')  
plt.xlabel('Sales Channel')                    
plt.ylabel('Avg Profit Margin (%)')            
for i,v in enumerate(apc.values):
    ax.text(i,v-2,f"{v:.2f}%",ha='center',fontweight='semibold')
plt.tight_layout()
plt.show()


# In[96]:


region_sales = (
    df
    .groupby('us_region')['revenue']
    .sum()
    .sort_values(ascending=False)  # so bars go top→bottom
    / 1e6                         # convert to millions
)

# 2. Plot
plt.figure(figsize=(10, 4))
sns.barplot(
    x=region_sales.values,
    y=region_sales.index,
    palette='Reds_r'          # dark→light green
)

# 3. Formatting
plt.title('Total Sales by US Region', fontsize=16, pad=12)
plt.xlabel('Total Sales (in Millions USD)', fontsize=12)
plt.ylabel('US Region', fontsize=12)
plt.xticks(rotation=0)
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.show()


# In[99]:


state_rev=(df.groupby('state_name').agg(revenue=('revenue', 'sum'),orders=('order_number', 'nunique'))
                                   .sort_values('revenue',ascending=False).head(10))
plt.figure(figsize=(15, 4))
sns.barplot(
    x=state_rev.index,
    y=state_rev['revenue']/1e6,
    palette='magma',)
plt.title('Top 10 states by Revenue',fontsize=14)
plt.xlabel('State',fontsize=12)
plt.ylabel('Revenue (in millions)',fontsize=12)
plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 4))
sns.barplot(
    x=state_rev.index,
    y=state_rev['orders'],
    palette='magma',)
plt.title('Top 10 states by Order Count',fontsize=14)
plt.xlabel('State',fontsize=12)
plt.ylabel('Order Count',fontsize=12)
plt.tight_layout()
plt.show()


# In[100]:


cust_summary = df.groupby('customer_name').agg(
    total_revenue=('revenue', 'sum'),
    total_profit=('profit', 'sum'),
    avg_margin=('profit_margin_pct', 'mean'),
    orders=('order_number', 'nunique')
)

# Convert revenue to millions
cust_summary['total_revenue_m'] = cust_summary['total_revenue'] / 1e6

plt.figure(figsize=(7, 5))

# Bubble chart with revenue in millions
sns.scatterplot(
    data=cust_summary,
    x='total_revenue_m',        # <-- use revenue in millions
    y='avg_margin',
    size='orders',
    sizes=(20, 200),
    alpha=0.7
)

plt.title('Customer Segmentation: Revenue vs. Profit Margin')
plt.xlabel('Total Revenue (Million USD)')  # <-- updated label
plt.ylabel('Avg Profit Margin (%)')
plt.tight_layout()
plt.show()


# In[102]:


num_cols = ['quantity', 'unit_price', 'revenue', 'cost', 'profit']

# Calculate the correlation matrix for these numeric features
corr = df[num_cols].corr()

# Set the figure size for clarity
plt.figure(figsize=(6,4))

# Plot the heatmap with annotations and a viridis colormap
sns.heatmap(
    corr,           # Data: correlation matrix
    annot=True,     # Display the correlation coefficients on the heatmap
    fmt=".2f",      # Format numbers to two decimal places
    cmap='RdYlGn'  # Color palette for the heatmap
)

# Add title for context
plt.title('Correlation Matrix')

# Adjust layout to prevent clipping
plt.tight_layout()

# Display the heatmap
plt.show()


# In[103]:


df


# In[104]:


df.to_csv('Sales_data(EDA).csv', index=False)


# In[ ]:





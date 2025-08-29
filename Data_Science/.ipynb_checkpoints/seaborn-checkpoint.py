# Installation commands
# pip3 install seaborn
# conda install seaborn

# Importing libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Accessing built-in datasets
sns.get_dataset_names()
iris_df = sns.load_dataset('iris')
tips_df = sns.load_dataset('tips')
titanic_df = sns.load_dataset('titanic')
exercise_df = sns.load_dataset('exercise')

# Data frame operations
iris_df.head()

# Univariate Distribution Plots (Distplot)
sns.histplot(iris_df['petal_length'])
sns.histplot(iris_df['petal_length'], kde=False)
sns.histplot(iris_df['petal_length'], stat='density', kde=True)

# Bivariate Distribution Plots (Jointplot)
# Scatter plot with histograms
sns.jointplot(x='petal_length', y='petal_width', data=iris_df)

# Hex bin plot
sns.jointplot(x='petal_length', y='petal_width', data=iris_df, kind='hex')

# KDE plot
sns.jointplot(x='petal_length', y='petal_width', data=iris_df, kind='kde')

# Pairwise Relationship Plots (Pairplot)
sns.set_style('ticks')
sns.pairplot(iris_df, hue='species', kind='scatter', diag_kind='kde', palette='husl')

# Categorical Scatter Plots
# Stripplot
sns.stripplot(x='species', y='petal_length', data=iris_df)
sns.stripplot(x='species', y='petal_length', data=iris_df, jitter=False)

# Swarmplot
sns.swarmplot(x='species', y='petal_length', data=iris_df)

# Box Plots
# Vertical Boxplot
sns.boxplot(x='species', y='petal_length', data=iris_df)

# Horizontal Boxplot (Wide Form Data)
sns.boxplot(data=iris_df, orient='h')

# Violin Plots
sns.violinplot(x='day', y='total_bill', data=tips_df)
sns.violinplot(x='day', y='total_bill', data=tips_df, hue='sex')

# Bar Plots
sns.barplot(x='sex', y='survived', data=titanic_df, hue='class')

# Count Plots
sns.countplot(data=titanic_df, x='class')
sns.countplot(data=titanic_df, x='class', palette='Greens')

# Point Plots
sns.pointplot(x='sex', y='survived', data=titanic_df, hue='class')

# Factor Plots (now catplot in newer versions)
# Point plot kind
sns.catplot(x='time', y='pulse', data=exercise_df, kind='point')

# Violin plot kind
sns.catplot(x='time', y='pulse', data=exercise_df, kind='violin')

# With hue parameter
sns.catplot(x='time', y='pulse', data=exercise_df, kind='violin', hue='diet')

# Count plot kind
sns.catplot(data=titanic_df, x='deck', kind='count', hue='deck')

# Linear Relationship Plots
# Regression plot (regplot)
sns.regplot(x='total_bill', y='tip', data=tips_df)

# Linear model plot (lmplot)
sns.lmplot(x='total_bill', y='tip', data=tips_df)
sns.lmplot(x='size', y='tip', data=tips_df)

# Facet Grid
g = sns.FacetGrid(tips_df, col='time')
g.map(plt.hist, 'tip')

# Pair Grid
g = sns.PairGrid(tips_df)
g.map(plt.scatter)
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot, cmap=plt.cm.Blues)
g.map_diag(sns.kdeplot, legend=False, lw=3)
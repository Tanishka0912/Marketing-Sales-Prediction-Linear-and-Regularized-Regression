# Marketing Sales Prediction: Linear and Regularized Regression

**Can we predict how much a business will sell based on how it markets its products?**

This project answers that question with regression models. It uses marketing spend, discounts, website traffic, and customer and product details to predict **Sales**, and explains which factors matter most.

---

## The Problem Statement

A company spends money on TV ads, social media, and search engine marketing. It also offers discounts and attracts website visitors. The goal is to understand:

- **Prediction:** given a set of marketing inputs, how many sales should we expect?
- **Explanation:** which inputs have the biggest effect on sales?

This is a **regression** problem, meaning we predict a number (Sales) rather than a category.

---

## The Dataset

The file `market_sales.csv` has **2,500 rows** (each row is one sales record) and **10 columns**. Sales average about **83,900**, ranging from about 18,700 to 221,000.

| Column | Type | What it means |
|---|---|---|
| `TV_Ad_Spend` | Number | Money spent on TV advertising |
| `Social_Media_Spend` | Number | Money spent on social media ads |
| `Discount_Offers` | Number | Discount level offered (0 to 30) |
| `Search_Engine_Marketing_Spend` | Number | Money spent on search engine ads |
| `Website_Traffic` | Number | Number of website visitors |
| `Conversion_Rate` | Number | Percentage of visitors who turn into buyers |
| `Product_Category` | Category | Electronics, Fashion, FMCG, Sports, Home Appliances |
| `Region` | Category | North, South, East, West |
| `Customer_Segment` | Category | Budget, Mid-tier, Premium |
| `Sales` | Number (**target**) | The value we want to predict |

---

## How the Project Works (Step by Step)

The notebook is split into two parts. Each step below explains **what** is done and **why**.

### Part 1: Linear Regression

| Step | What we do | Why we do it |
|---|---|---|
| **1. Check for missing data** | Found gaps in 5 columns (3.8% to 7.3% missing per column) | Models can't learn from blank cells |
| **2. Fill the gaps** | Spend columns filled with the **average**; `Product_Category` with the **most common value**; `Discount_Offers` and `Region` with **random samples** that keep the original mix | Keeps all 2,500 rows instead of throwing data away. A random seed makes the result repeatable |
| **3. Check for duplicates** | Found none | Duplicate rows would give some records extra weight |
| **4. Explore the data (EDA)** | Charts for distributions, averages per category, scatter plots against Sales, and a correlation heatmap | Shows patterns and problems before modeling |
| **5. Encode categories** | Turn text columns into 0/1 columns (one-hot encoding) | Models only understand numbers |
| **6. Split the data** | 70% for training, 30% for testing | The model learns on one part, and we test it on data it has never seen, which is an honest check |
| **7. Train a baseline** | Fit a Linear Regression model | Gives a simple, explainable starting point |
| **8. Check assumptions** | Test whether linear regression is a fair choice for this data (see below) | A model can score well and still be unreliable if its assumptions are broken |

**The four assumption checks**

| Assumption | Plain meaning | Result |
|---|---|---|
| **Linearity** | Sales rises or falls in a roughly straight line with each input | Checked with scatter plots |
| **No multicollinearity** | Inputs aren't near-copies of each other. Measured with **VIF** (under 5 is fine) | All feature VIFs are below 2, so this is fine |
| **Constant error spread** | The model is about equally accurate for small and large sales. Tested with **Breusch-Pagan** | p = 0.099, above 0.05, so no strong evidence of a problem |
| **No pattern in errors over rows** | Errors aren't linked to the row order. Tested with **Durbin-Watson** (near 2 is ideal) | 1.99, so this is fine |

### Part 2: Regularized Regression

| Step | What we do | Why we do it |
|---|---|---|
| **1. Scale the features** | Put every input on a similar scale (the scaler learns from training data only) | Regularization penalizes large numbers, so a feature measured in thousands would be unfairly punished compared to one measured in single digits |
| **2. Train Ridge, Lasso, ElasticNet** | Three "regularized" versions of linear regression | They discourage extreme coefficients, which makes models steadier |
| **3. Tune with K-Fold cross-validation** | Split the training data into 5 parts and rotate which part is used for checking | One train/test split can be lucky or unlucky. Averaging over 5 gives a more reliable score |
| **4. Compare and pick a winner** | Choose the model with the lowest average error (RMSE) in cross-validation | Gives an objective way to choose |
| **5. Inspect coefficients** | See how each model weighted each feature | Shows whether any feature was dropped or just shrunk |

---

## Results Explained

### Test-set performance (data the model never saw)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Lasso (tuned, alpha = 1.0) | 5,190.37 | 6,758.33 | 0.9411 |
| ElasticNet (tuned) | 5,190.26 | 6,758.51 | 0.9411 |
| Ridge (tuned) | 5,190.33 | 6,758.52 | 0.9411 |
| Linear Regression (baseline) | 5,190.43 | 6,758.53 | 0.9411 |

### What these numbers mean

- **R² = 0.94:** the model explains about **94% of the ups and downs in Sales**. A score of 1.0 would be perfect and 0 would be no better than always guessing the average.
- **MAE = about 5,190:** on average, a prediction is off by roughly 5,190 sales units. Against an average sale value of about 83,900, that is around **6%**.
- **RMSE = about 6,760:** similar to MAE, but it punishes big misses more. Because it is a bit higher than MAE, a few predictions are further off than the typical one.
- **Train R² (0.945) vs test R² (0.941):** nearly identical, so the model is **not overfitting**. It learned real patterns rather than memorizing the training data.

### Cross-validation (more reliable than a single split)

Average RMSE across 5 folds is about **6,598**, with a standard deviation of about 207. A small spread means performance doesn't swing much depending on which rows are used.

---

## What Drives Sales? (Interpreting the Model)

The baseline model gives each feature a **coefficient**: the change in Sales linked to a **one-unit increase** in that feature, with everything else held constant.

| Feature | Coefficient | In plain English |
|---|---|---|
| `Conversion_Rate` | +36,086 | Each extra percentage point of conversion goes with about 36,000 more sales. **By far the biggest driver** |
| `Discount_Offers` | +506 | Each extra unit of discount goes with about 506 more sales |
| `Search_Engine_Marketing_Spend` | +2.18 | Each extra unit spent on search ads goes with about 2.18 more sales |
| `Social_Media_Spend` | +1.71 | Each extra unit spent on social media goes with about 1.71 more sales |
| `TV_Ad_Spend` | +1.35 | Each extra unit spent on TV goes with about 1.35 more sales |
| `Website_Traffic` | -0.54 | Tiny effect, and slightly negative |

**Categories are compared against a reference group** (the first one alphabetically, which was dropped during encoding):

- **Product category** (vs Electronics): Fashion is highest at about +989, then Sports (+871), Home Appliances (+493) and FMCG (+293).
- **Region** (vs East): South is about 617 lower; North and West are close to East.
- **Customer segment** (vs Budget): Mid-tier (+43) and Premium (+19) are almost the same as Budget.

**Which features matter most?** The scaled (Part 2) coefficients compare features fairly. Ranked by importance, they are Conversion Rate, Search Engine Spend, Social Media Spend, TV Spend, then Discounts. Region, segment and category effects are small.

> **Note:** these show **association**, not proof of cause. For example, a higher conversion rate goes with higher sales, but this data alone can't prove that raising the conversion rate *causes* more sales.

---

## Why Use Regularization If Results Are the Same?

Ridge, Lasso, and ElasticNet scored almost exactly the same as plain Linear Regression. That is a **useful finding**, not a failure:

- The data is clean, the relationships are mostly linear, and the features are not tangled together (low VIF), so plain linear regression already works very well.
- Lasso did **not** remove any feature, which suggests every input adds at least something.
- Lasso was chosen as the final model because it had the lowest cross-validation error, but the gap between models is tiny. In practice, plain Linear Regression is equally valid and the easiest to explain.

Regularization becomes much more valuable when there are many features, features overlap heavily, or there is less data.

---

## Limitations

- **Missing values were filled with simple methods.** Random filling adds a little noise, and mean filling shrinks natural variation.
- **Outliers were not treated.** Box plots were drawn, but extreme values were not removed or capped.
- **Results are linked to this dataset only.** The model may not carry over to other markets or time periods.
- **Association is not causation**, as noted above.
- **The `Discount_Offers` fill used only 10 and 15**, although the column has 7 distinct values, which slightly changes its distribution.

---

## Glossary

| Term | Meaning |
|---|---|
| **Regression** | Predicting a number from other numbers |
| **Target** | The thing we want to predict (Sales) |
| **Feature** | An input used for prediction (e.g. TV spend) |
| **Coefficient** | How much the prediction changes when a feature goes up by one unit |
| **Overfitting** | Memorizing training data so the model does poorly on new data |
| **R²** | Share of the variation in Sales the model explains (0 to 1) |
| **MAE** | Average size of the prediction errors |
| **RMSE** | Like MAE, but big errors count more |
| **One-hot encoding** | Turning a category into separate 0/1 columns |
| **VIF** | Measures how much a feature overlaps with the others |
| **Regularization** | Penalizing large coefficients to keep the model stable |
| **Ridge (L2)** | Shrinks coefficients toward zero but keeps all of them |
| **Lasso (L1)** | Can shrink some coefficients all the way to zero (feature selection) |
| **ElasticNet** | A mix of Ridge and Lasso |
| **Cross-validation** | Testing several times on different slices of data for a more reliable score |

---

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn scipy jupyter
```

### Run the notebook

Make sure the dataset is named `market_sales.csv` and sits next to the notebook, then run:

```bash
jupyter notebook Project_-_Marketing_Sales_Prediction_Linear_and_Regularized_Regression.ipynb
```

Run the cells from top to bottom. A cleaned copy of the data is saved as `market_sales_nv_cleaned.csv`.

---

## Project Structure

```
your-repo/
├── Project_-_Marketing_Sales_Prediction_Linear_and_Regularized_Regression.ipynb
├── market_sales.csv
└── README.md
```

---

## Tech Stack

Python, pandas, NumPy, matplotlib, seaborn, statsmodels, scikit-learn, SciPy, Jupyter

## License

MIT License. Add a `LICENSE` file to the repository to use this.

## Author

Your Name: [@your-username](https://github.com/your-username)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from wordcloud import WordCloud
from fpdf import FPDF
from datetime import datetime
from sklearn.decomposition import PCA
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import zscore

# Grafik kayıt fonksiyonu
def save_plot(fig, path):
    fig.savefig(path, bbox_inches='tight')


# Eksik verileri analiz et
def analyze_missing_data(df):
    missing = df.isnull().sum()
    if missing.sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    missing[missing > 0].sort_values().plot(kind='barh', ax=ax, color='salmon')
    ax.set_title('Missing Values Count per Column')
    title = "Missing Values Count"
    explanation = "Shows how many missing values are present in each column. Useful for identifying data quality issues."
    return (title, explanation, fig)


def plot_missing_heatmap(df):
    if df.isnull().sum().sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='Reds', ax=ax)
    ax.set_title("Missing Values Heatmap")
    title = "Missing Values Heatmap"
    explanation = "Visualizes missing values across the dataset. Red areas indicate null entries."
    return (title, explanation, fig)


# Zaman serisi trend analizi
def analyze_trends(df):
    date_cols = df.select_dtypes(include=['datetime64', 'object']).columns
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            numeric = df.select_dtypes(include=np.number)
            if numeric.shape[1] > 0:
                fig, ax = plt.subplots(figsize=(10, 4))
                numeric.iloc[:, 0].resample('D').mean().plot(ax=ax)
                ax.set_title(f"Trend Over Time ({numeric.columns[0]})")
                title = "Time Series Trend"
                explanation = "Displays average values over time to detect trends or seasonality."
                return (title, explanation, fig)
        except:
            continue
    return None


# Kategorik analiz
def analyze_categorical(df):
    figures = []
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        vc = df[col].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=vc.values, y=vc.index, ax=ax, palette='viridis')
        ax.set_title(f"Top Values in {col}")
        title = f"Top Values in {col}"
        explanation = f"Shows the most frequent values in column '{col}', useful for detecting dominant categories."
        figures.append((title, explanation, fig))
    return figures


def analyze_value_counts(df):
    figures = []
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(6, 3))
        df[col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f"Value Counts for {col}")
        title = f"Value Counts for {col}"
        explanation = f"Breaks down value frequencies for column '{col}', revealing how balanced the data is."
        figures.append((title, explanation, fig))
    return figures


# Sayısal dağılım
def analyze_numeric_distributions(df):
    figures = []
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        title = f"Distribution of {col}"
        explanation = f"Shows how the values of '{col}' are distributed. Helps spot skewed or normal distributions."
        figures.append((title, explanation, fig))
    return figures


# Boxplot
def analyze_boxplots(df):
    figures = []
    for col in df.select_dtypes(include=np.number).columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
        title = f"Boxplot of {col}"
        explanation = f"Displays distribution, median, and potential outliers in '{col}'."
        figures.append((title, explanation, fig))
    return figures


# Korelasyon matrisi
def analyze_correlations(df):
    corr = df.select_dtypes(include=np.number).corr()
    if corr.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    title = "Correlation Matrix"
    explanation = "Displays how numerical variables are related. Strong correlations may imply dependencies."
    return (title, explanation, fig)


# Skewness-Kurtosis
def analyze_skew_kurt(df):
    num = df.select_dtypes(include=np.number)
    return pd.DataFrame({
        'Skewness': num.skew(),
        'Kurtosis': num.kurtosis()
    })


# Aykırı değer tespiti
def detect_outliers(df):
    outliers = {}
    for col in df.select_dtypes(include=np.number).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        out = df[(df[col] < lower) | (df[col] > upper)]
        if not out.empty:
            outliers[col] = len(out)
    return pd.DataFrame.from_dict(outliers, orient='index', columns=['Outlier Count'])


# Anomali (Z-score)
def detect_anomalies(df):
    from scipy.stats import zscore
    numeric = df.select_dtypes(include=np.number)
    z_scores = np.abs(zscore(numeric))
    anomaly = (z_scores > 3).any(axis=1)
    return df[anomaly]


# Text column analiz
def analyze_text_columns(df):
    result = {}
    for col in df.select_dtypes(include='object'):
        if df[col].apply(lambda x: isinstance(x, str)).mean() > 0.8:
            combined = " ".join(df[col].dropna().astype(str))
            wc = WordCloud(width=600, height=300, background_color='white').generate(combined)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            save_plot(fig, f"wordcloud_{col}.png")
            result[col] = {
                'Unique Words': len(set(combined.split())),
                'WordCloud': f"wordcloud_{col}.png"
            }
    return result


# PCA görselleştirme
def analyze_pca(df):
    numeric = df.select_dtypes(include=np.number).dropna()
    if numeric.shape[1] < 2:
        return None
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(numeric)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    ax.set_title("PCA Projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    title = "PCA Projection"
    explanation = "Visualizes high-dimensional data in 2D using PCA. Helps identify clusters and structure."
    return (title, explanation, fig)


# Scatter Matrix (pairplot alternatifi)
def analyze_scatter_matrix(df):
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return None
    fig = scatter_matrix(numeric, figsize=(8, 8), diagonal='kde')
    title = "Scatter Matrix"
    explanation = "Shows pairwise relationships between numerical features. Useful for spotting linear correlations."
    return (title, explanation, plt.gcf())


# PDF oluşturucu sınıf
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 16)
        self.cell(0, 10, "Smart Data Report", ln=True, align='C')
        self.ln(10)


def generate_pdf_report(df, chart_paths, output_file):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}")

    for title, path in chart_paths:
        if os.path.exists(path):
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, title, ln=True)
            pdf.image(path, x=10, y=30, w=180)

    pdf.output(output_file)
    return output_file

def generate_smart_insights(df):
    insights = []
    # Eksik veri oranı
    missing_ratio = df.isnull().mean()
    high_missing = missing_ratio[missing_ratio > 0.5]
    if not high_missing.empty:
        insights.append(f"⚠️ High missing data in: {', '.join(high_missing.index)}")

    # Çok yüksek korelasyonlar
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        corr_matrix = numeric_df.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr = corr_matrix.max().max()
        if max_corr > 0.85:
            col1, col2 = np.unravel_index(np.argmax(corr_matrix.values), corr_matrix.shape)
            insights.append(f"📌 Strong correlation between: {corr_matrix.columns[col1]} and {corr_matrix.columns[col2]} (r={max_corr:.2f})")

    # Yinelenen satırlar
    if df.duplicated().sum() > 0:
        insights.append(f"♻️ Duplicate rows detected: {df.duplicated().sum()}")

    return insights


def detect_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number])
    outlier_info = {}
    for col in numeric_cols.columns:
        z_scores = zscore(df[col].dropna())
        outliers = np.sum(np.abs(z_scores) > 3)
        if outliers > 0:
            outlier_info[col] = outliers
    return outlier_info


def get_feature_importance(df, target_col):
    df = df.dropna()
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categoricals
    X = X.copy()
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    model_type = RandomForestClassifier if y.nunique() <= 10 else RandomForestRegressor
    model = model_type(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    importance_dict = dict(zip(X.columns, importances))
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def predict_from_input(df, target_col, input_dict):
    df = df.dropna()
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categoricals
    X = X.copy()
    label_encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    model_type = RandomForestClassifier if y.nunique() <= 10 else RandomForestRegressor
    model = model_type(n_estimators=100, random_state=42)
    model.fit(X, y)

    input_df = pd.DataFrame([input_dict])
    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)[0]
    return prediction

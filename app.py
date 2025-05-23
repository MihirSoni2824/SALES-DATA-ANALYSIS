import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import gradio as gr

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

def analyze_sales(csv_path):
    try:
        # Read the uploaded CSV with proper encoding for Excel-exported files
        df = pd.read_csv(csv_path, encoding='ISO-8859-1')

        report = "DATA ANALYSIS REPORT\n\n"
        report += f"Columns: {list(df.columns)}\n"
        report += f"Total records before cleaning: {len(df)}\n"

        # Drop duplicates
        df.drop_duplicates(inplace=True)
        report += f"After dropping duplicates: {len(df)} rows\n"

        # Drop missing values
        df.dropna(inplace=True)
        report += f"After dropping missing values: {len(df)} rows\n"

        # Detect and convert date column
        date_col_candidates = [col for col in df.columns if 'Date' in col or 'date' in col]
        date_col = None
        if date_col_candidates:
            date_col = date_col_candidates[0]
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                report += f"Converted '{date_col}' to datetime format.\n"
            except Exception as e:
                report += f"Could not convert '{date_col}' to datetime: {e}\n"
                date_col = None
        else:
            report += "No date column found for time series analysis.\n"

        # Plot 1: Sales over time
        if date_col and 'Sales' in df.columns:
            sales_over_time = df.groupby(date_col)['Sales'].sum().sort_index()
            plt.figure(figsize=(6, 4))
            plt.plot(sales_over_time.index, sales_over_time.values, color='cyan', marker='o')
            plt.title("Sales Over Time")
            plt.xlabel("Date")
            plt.ylabel("Total Sales")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("output/sales_over_time.png")
            plt.close()

        # Plot 2: Profit vs Discount
        if 'Profit' in df.columns and 'Discount' in df.columns:
            plt.figure(figsize=(6, 4))
            plt.scatter(df['Discount'], df['Profit'], alpha=0.5, color='magenta')
            plt.title("Profit vs Discount")
            plt.xlabel("Discount")
            plt.ylabel("Profit")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("output/profit_vs_discount.png")
            plt.close()

        # Plot 3: Sales by Region
        if 'Region' in df.columns and 'Sales' in df.columns:
            region_sales = df.groupby('Region')['Sales'].sum()
            region_sales.plot(kind='bar', color='skyblue', figsize=(6, 4))
            plt.title("Sales by Region")
            plt.ylabel("Total Sales")
            plt.tight_layout()
            plt.savefig("output/sales_by_region.png")
            plt.close()

        # Plot 4: Sales by Category
        if 'Category' in df.columns and 'Sales' in df.columns:
            category_sales = df.groupby('Category')['Sales'].sum()
            category_sales.plot(kind='bar', color='orange', figsize=(6, 4))
            plt.title("Sales by Category")
            plt.ylabel("Total Sales")
            plt.tight_layout()
            plt.savefig("output/sales_by_category.png")
            plt.close()

        # Linear Regression
        if all(col in df.columns for col in ['Profit', 'Discount', 'Sales']):
            X = df[['Profit', 'Discount']]
            y = df['Sales']
            model = LinearRegression()
            model.fit(X, y)
            preds = model.predict(X)
            r2 = r2_score(y, preds)
            mse = mean_squared_error(y, preds)
            report += "\nLinear Regression Results:\n"
            report += f"  R² Score: {r2:.4f}\n"
            report += f"  Mean Squared Error: {mse:.2f}\n"
        else:
            report += "Not enough data to perform regression analysis.\n"

        # Save text report
        with open("output/metrics.txt", "w") as f:
            f.write(report)

        # Create ZIP of all outputs
        import zipfile
        zip_path = "output/analysis_results.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in os.listdir("output"):
                if file != "analysis_results.zip":
                    zipf.write(os.path.join("output", file), arcname=file)

        return report, zip_path

    except Exception as e:
        return f"ERROR: {str(e)}", None

# Gradio interface with dark abstract theme
css = """.gradio-container {background: linear-gradient(to right, #0f2027, #203a43, #2c5364); color: white;}"""
theme = gr.themes.Ocean()

with gr.Blocks(css=css, theme=theme) as demo:
    gr.Markdown("## Sales Performance Analyzer")
    gr.Markdown("Upload a sales CSV file to generate automated EDA, charts, and predictions.")

    file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
    output_text = gr.Textbox(label="Analysis Report", lines=20)
    output_file = gr.File(label="Download Results ZIP")
    submit_btn = gr.Button("Run Analysis")

    def run(file):
        return analyze_sales(file.name)

    submit_btn.click(run, inputs=file_input, outputs=[output_text, output_file])

demo.launch()

import pandas as pd
import plotly.express as px


def label_distribution(data: pd.DataFrame):
    label_counts = data["label"].value_counts().reset_index()
    label_counts.columns = ['Rating', 'Count']
    label_counts['Rating'] = label_counts['Rating'].map({0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High'})
    fig = px.bar(
        label_counts,
        x='Rating',
        y='Count',
        title='Label Distribution',
        labels={'Rating': 'Label', 'Count': 'Count'},
        color='Rating',  # Apply colors to each category
        color_discrete_map={'Low': 'blue', 'Medium': 'green', 'High': 'orange', 'Very High': 'red'},  # Specify colors
    )

    # Customize layout and appearance
    fig.update_layout(
        xaxis_title="Label",
        yaxis_title="Count",
        legend_title="Label Category",
        font=dict(size=12),
    )

    fig.show()


def repay_percentage(data: pd.DataFrame):
    intervals = list(range(0, 101, 10))
    fig = px.histogram(data,
                       x=data["loan_repay_percentage"] * 100,
                       nbins=len(intervals),
                       labels={'loan_repay_percentage': 'Repay Percentage'},
                       title='Loan Reimbursement Percentage Distribution',
                       category_orders={"reimbursement_percentage": [f"{i}-{i + 9}%" for i in intervals[:-1]]},
                       )
    fig.update_layout(
        xaxis_title="Repay Percentage",
        yaxis_title="Count",
        legend_title=None,  # No legend title
        font=dict(size=12),
    )
    fig.update_xaxes(tickvals=list(range(0, 101, 5)))
    fig.show()


def zero_loans(df: pd.DataFrame):
    total_num = df[df['loan_repay_percentage'] == 0].shape[0]
    loaned_once_num = df[(df['loan_repay_percentage'] == 0) & (df['numLoans'] > 0)].shape[0]

    labels = ['Repaid 0% with at least 1 loan taken', 'Took 0 Loans']
    values = [loaned_once_num, total_num - loaned_once_num]

    fig = px.pie(
        names=labels,
        values=values,
        title='Loaners Repaid 0% and No Loans Distribution',
    )

    fig.show()


def analyze_data(data: pd.DataFrame):
    # label_distribution(data)
    repay_percentage(data)
    # zero_loans(data)

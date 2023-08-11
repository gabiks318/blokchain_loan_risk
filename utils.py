import datetime

import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score

from constants import columns_to_drop, str_columns


def split_data():
    pass


def add_labels(data: dict, num_of_days: int):
    positive_labels = 0
    negative_labels = 0
    for user_id in data.keys():
        first_debt_date = None
        last_debt_date = None
        for day in data[user_id]['Calendar']:
            if data[user_id]['Calendar'][day]['sizeDebtUSD'] > 0:
                first_debt_date = day
                break
        for day in reversed(data[user_id]['Calendar']):
            if data[user_id]['Calendar'][day]['sizeDebtUSD'] > 0:
                last_debt_date = day
                break
        if first_debt_date is None:
            data[user_id]['label'] = 0
            negative_labels += 1
        else:
            if last_debt_date is None:
                last_debt_date = list(reversed(data[user_id]['Calendar']))[0]
            # Convert to datetime and find difference
            first_debt_date = pd.to_datetime(first_debt_date)
            last_debt_date = pd.to_datetime(last_debt_date)
            # print((last_debt_date - first_debt_date).days)
            label = int((last_debt_date - first_debt_date).days > num_of_days)
            data[user_id]['label'] = label
            if label == 1:
                positive_labels += 1
            else:
                negative_labels += 1
    print(
        f"Positive labels: {positive_labels}\nNegative labels: {negative_labels}\nRatio: {100 * positive_labels / (negative_labels + positive_labels)}%")
    return data


def __split_date_list(date_string: str):
    date_list = date_string.split(' ')
    return [d for d in date_list if (d != '' and ':' not in d)]


def classify_by_percentage(percentage: float) -> int:
    class_ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]
    # class_ranges = [(0, 0.5), (0.5, 1)]
    for i, label in enumerate(class_ranges):
        if label[0] <= percentage <= label[1]:
            return i
    return 0


def add_labels_v2(data: dict, num_of_days: int):
    users_to_remove = []
    for user_id in data:
        loan_dates = __split_date_list(data[user_id]['datesLoan'])
        total_loans = len(loan_dates)
        repaid_loans = 0
        user_data = data[user_id]
        loans = []
        # Find amount and date of each loan
        for loan_date_str in loan_dates:
            if loan_date_str not in user_data['Calendar']:
                if len(user_data['Calendar']) > 0:
                    users_to_remove.append(user_id)
                    break
                else:
                    repay_dates = __split_date_list(user_data['datesReimbursement'])
                    if len(repay_dates) == 1 and repay_dates[0] == loan_date_str:
                        user_data['label'] = 1
                        break
                    else:
                        users_to_remove.append(user_id)
                        break
            else:
                current_day_debt = user_data['Calendar'][loan_date_str]['sizeDebtUSD']
            previous_date_str = (datetime.datetime.strptime(loan_date_str, "%Y-%m-%d") - datetime.timedelta(days=1)) \
                .strftime('%Y-%m-%d')

            if previous_date_str not in user_data['Calendar']:
                loan_amount = current_day_debt
            else:
                loan_amount = current_day_debt - user_data['Calendar'][previous_date_str]['sizeDebtUSD']
            loans.append([loan_date_str, loan_amount])

        for day in user_data['Calendar']:
            repay_amount = user_data['Calendar'][day]['sizeDebtUSD']
            if repay_amount > 0 and len(loans) > 0:
                current_loan = loans[0]
                while current_loan[1] < repay_amount:
                    repay_amount -= current_loan[1]
                    # Find how much time it took to repay the loan
                    if datetime.datetime.strptime(day, "%Y-%m-%d") - datetime.datetime.strptime(current_loan[0],
                                                                                                "%Y-%m-%d") <= datetime.timedelta(
                        days=num_of_days):
                        repaid_loans += 1
                    loans.pop(0)
                    if len(loans) == 0:
                        break
                    current_loan = loans[0]
        user_data['loan_repay_percentage'] = repaid_loans / total_loans if total_loans > 0 else 0
        user_data['label'] = classify_by_percentage(user_data['loan_repay_percentage'])

    for user_id in users_to_remove:
        del data[user_id]
    return data


def __get_length(x: str):
    return len([i for i in x.split(" ") if i != ""])


def __get_first(x: str):
    ls = [i for i in x.split(" ") if i != ""]
    return 0 if len(ls) == 0 else ls[0]


def __get_last(x: str):
    ls = [i for i in x.split(" ") if i != ""]
    return 0 if len(ls) == 0 else ls[-1]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df['numLoans'] = df['timeStampsLoans'].apply(__get_length)
    df["firstLoanTimestamp"] = df['timeStampsLoans'].apply(__get_first)
    df["lastLoanTimestamp"] = df['timeStampsLoans'].apply(__get_last)
    df['numReimbursements'] = df['timeStampsReimbursement'].apply(__get_length)
    df["firstReimbursementTimestamp"] = df['timeStampsReimbursement'].apply(__get_first)
    df["lastReimbursementTimestamp"] = df['timeStampsReimbursement'].apply(__get_last)
    df['numSupplyCollateral'] = df['timeStampsSupplyCollateral'].apply(__get_length)
    df["firstSupplyCollateralTimestamp"] = df['timeStampsSupplyCollateral'].apply(__get_first)
    df["lastSupplyCollateralTimestamp"] = df['timeStampsSupplyCollateral'].apply(__get_last)
    df['numWithdrawCollateral'] = df['timeStampsWithdrawCollateral'].apply(__get_length)
    df["firstWithdrawCollateralTimestamp"] = df['timeStampsWithdrawCollateral'].apply(__get_first)
    df["lastWithdrawCollateralTimestamp"] = df['timeStampsWithdrawCollateral'].apply(__get_last)

    # for row in df.iterrows():
    #     loans_timestamps = row[1]["timeStampsLoans"].split(" ")
    #     reimbursements_timestamps = row[1]["timeStampsReimbursement"].split(" ")
    #     collaterals_timestamps = row[1]["timeStampsSupplyCollateral"].split(" ")
    #     collateral_withdraw_timestamps = row[1]["timeStampsWithdrawCollateral"].split(" ")
    #     print(
    #         f"loans: {len(loans_timestamps)}, reimbursements: {len(reimbursements_timestamps)}, supply: {len(collaterals_timestamps)}, withdraw: {len(collateral_withdraw_timestamps)}")
    return df.drop(columns=['timeStampsLoans', 'timeStampsReimbursement', 'timeStampsSupplyCollateral',
                            'timeStampsWithdrawCollateral'])


def etl(data: dict):
    del data['generalStats']
    labeled_data = add_labels_v2(data, 200)
    df = (pd.DataFrame.from_dict(labeled_data, orient='index')
          .reset_index()
          .rename(columns={"index": "user_id"})
          .assign(**{k: lambda x: x[k].astype('float') for k in str_columns})
          )
    df = extract_features(df).drop(columns=columns_to_drop)

    return df


def evaluate_model(model, train_data, validation_data):
    train_prediction = pd.Series(model.predict(train_data.drop(columns=['label', "loan_repay_percentage"]))).apply(classify_by_percentage)
    validation_prediction = pd.Series(model.predict(validation_data.drop(columns=['label', "loan_repay_percentage"]))).apply(classify_by_percentage)
    train_accuracy = accuracy_score(train_data['label'], train_prediction)
    validation_accuracy = accuracy_score(validation_data['label'], validation_prediction)
    print(f"Train accuracy: {train_accuracy}")
    print(f"Validation accuracy: {validation_accuracy}")


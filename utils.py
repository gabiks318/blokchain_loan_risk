import pandas as pd

from constants import columns_to_drop, str_columns


def split_data():
    pass


def add_labels(df: pd.DataFrame):
    users = [df['user_id'].loc[0], ]
    df['label'] = df['user_id'].apply(lambda x: 1 if x in users else 0)
    return df.drop(columns=['user_id'])


def extract_features(df: pd.DataFrame):
    df['numLoans'] = df['timeStampsLoans'].apply(lambda x: len(x.split(" ")))
    df["firstLoanTimestamp"] = df['timeStampsLoans'].apply(lambda x: x.split(" ")[0])
    df["lastLoanTimestamp"] = df['timeStampsLoans'].apply(lambda x: x.split(" ")[-1])
    df['numReimbursements'] = df['timeStampsReimbursement'].apply(lambda x: len(x.split(" ")))
    df["firstReimbursementTimestamp"] = df['timeStampsReimbursement'].apply(lambda x: x.split(" ")[0])
    df["lastReimbursementTimestamp"] = df['timeStampsReimbursement'].apply(lambda x: x.split(" ")[-1])
    df['numSupplyCollateral'] = df['timeStampsSupplyCollateral'].apply(lambda x: len(x.split(" ")))
    df["firstSupplyCollateralTimestamp"] = df['timeStampsSupplyCollateral'].apply(lambda x: x.split(" ")[0])
    df["lastSupplyCollateralTimestamp"] = df['timeStampsSupplyCollateral'].apply(lambda x: x.split(" ")[-1])
    df['numWithdrawCollateral'] = df['timeStampsWithdrawCollateral'].apply(lambda x: len(x.split(" ")))
    df["firstWithdrawCollateralTimestamp"] = df['timeStampsWithdrawCollateral'].apply(lambda x: x.split(" ")[0])
    df["lastWithdrawCollateralTimestamp"] = df['timeStampsWithdrawCollateral'].apply(lambda x: x.split(" ")[-1])

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
    df = (pd.DataFrame.from_dict(data, orient='index')
          .reset_index()
          .rename(columns={"index": "user_id"})
          .drop(columns=columns_to_drop)
          .assign(**{k: lambda x: x[k].astype('float') for k in str_columns})
          )
    df = extract_features(df)

    df = add_labels(df)
    # Normalize
    return df

validation_dates = ['2022-11-23', '2022-11-24', '2022-11-25', '2022-11-26', '2022-11-27', '2022-11-28',
                    '2022-11-29', '2022-11-30', '2022-12-01']
test_dates = ['2022-12-02', '2022-12-03', '2022-12-04', '2022-12-05', '2022-12-06', '2022-12-07', '2022-12-08',
              '2022-12-09', '2022-12-10', '2022-12-11', '2022-12-12']
columns_to_drop = ['datesSupplyCollateral',
                   'datesLoan',
                   'datesReimbursement',
                   'datesWithdrawCollateral',
                   "Calendar",
                   "user_id",
                   'timeUTCFirstAnyTransactionAccount']
str_columns = [
    # 'user_id',
    # 'timeStampsSupplyCollateral',
    # 'timeStampsLoans',
    # 'timeStampsReimbursement',
    # 'timeStampsWithdrawCollateral',
    'timeStampFirstAnyTransactionAccount', ]


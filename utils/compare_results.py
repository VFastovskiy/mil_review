import pandas as pd

if __name__ == '__main__':
    res_test_1 = pd.read_csv('../fingerprints_exp_1/eval_results_exp_1/evaluation_results_exp_1_with_moe.csv', header=0)
    res_test_2 = pd.read_csv('../fingerprints_exp_2/eval_results_exp_2/evaluation_results_exp_2_with_moe.csv', header=0)
    merged_res = res_test_1.merge(res_test_2, how='outer', on='fingerprint')
    merged_res['diff%'] = ((merged_res['ba_test_2'] - merged_res['ba_test_1']) / merged_res['ba_test_1']) * 100
    merged_res['diff%'] = merged_res['diff%'].round(0)
    merged_res.sort_values(by='diff%', ascending=True, inplace=True)
    merged_res.to_csv('../results/evaluation_results_comparison.csv', index=False)
    print(merged_res)
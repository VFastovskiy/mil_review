import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np

def sorting(results):
    sorted_results_diff = results.sort_values(by=['diff%'], ascending=True).head(10)
    print(sorted_results_diff)
    sorted_results_ba = results.sort_values(by=['ba_test_1'], ascending=False).head(10)
    print(sorted_results_ba)

    print(results['type'].value_counts().get('3D', 0))


def plotting(basedir, results, test, fingerprint_order):
    top_performers = results[results['fingerprint'].isin(fingerprint_order)]
    top_performers = top_performers.set_index('fingerprint').loc[fingerprint_order].reset_index()

    plt.figure(figsize=(10, 6))
    plt.bar(
        top_performers['fingerprint'],
        top_performers[test],
        color=['blue' if t == '3D' else 'green' for t in top_performers['type']]
    )
    plt.xlabel('FP')
    plt.ylabel(test)
    plt.title(f'Top Performing 3D and 2D Fingerprints ({test})')
    plt.xticks(rotation=45, ha='right')

    plt.ylim(0.5, 1)
    blue_patch = mpatches.Patch(color='blue', label='3D')
    green_patch = mpatches.Patch(color='green', label='2D')
    plt.legend(handles=[blue_patch, green_patch], loc='upper right')
    plt.tight_layout()
    filename = f"results_{test}.png"
    plt.savefig(os.path.join(basedir, filename))


def stacked_plot(basedir, results, test_list, fingerprint_order):
    top_performers = results[results['fingerprint'].isin(fingerprint_order)]
    top_performers = top_performers.set_index('fingerprint').loc[fingerprint_order].reset_index()

    plt.figure(figsize=(12, 6))

    bottom = np.zeros(len(fingerprint_order))

    colors = ['blue', 'green']

    for i, test in enumerate(test_list):
        plt.bar(
            top_performers['fingerprint'],
            top_performers[test],
            bottom=bottom,
            color=colors[i],
            label=test
        )
        # update the bottom to the current height for stacking
        bottom += top_performers[test]

    plt.xlabel('Fingerprint')
    plt.ylabel('Test Value')
    plt.title('Stacked Comparison of Tests for Top Fingerprints')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.5, 1)
    plt.legend(loc='upper right')
    plt.tight_layout()
    filename = "results_stacked_tests.png"
    plt.savefig(os.path.join(basedir, filename))


def overlapping_plot(basedir, results, test_list, fingerprint_order):
    top_performers = results[results['fingerprint'].isin(fingerprint_order)]
    top_performers = top_performers.set_index('fingerprint').loc[fingerprint_order].reset_index()

    x = np.arange(len(fingerprint_order))  # locations for fingerprints
    bar_width = 0.35  # width of the bars

    plt.figure(figsize=(12, 6))

    # Plot bars for each test
    for i, test in enumerate(test_list):
        plt.bar(
            x + i * bar_width,
            top_performers[test],
            width=bar_width,
            label=f'{test}',
            color='blue' if test == 'ba_test_1' else 'green'
        )

    plt.xlabel('Fingerprint')
    plt.ylabel('Balanced Accuracy')
    plt.title('2D vs 3D Fingerprints')
    plt.xticks(x + bar_width / 2, fingerprint_order, rotation=45, ha='right')

    plt.ylim(0.5, 1)
    plt.legend(loc='upper right')
    plt.tight_layout()
    filename = "results_overlapping_tests.png"
    plt.savefig(os.path.join(basedir, filename))


if __name__ == '__main__':
    basedir = os.path.dirname(__file__)
    results = pd.read_csv(os.path.join(basedir, 'evaluation_results_comparison.csv'))

    sorting(results)

    test_list = ['ba_test_1', 'ba_test_2']

    df_3d = results[results['type'] == '3D'].nlargest(12, 'ba_test_1')
    df_2d = results[results['type'] == '2D'].nlargest(5, 'ba_test_1')
    top_fingerprints = pd.concat([df_3d, df_2d])['fingerprint'].unique()

    fingerprint_order = list(top_fingerprints)

    stacked_plot(basedir, results, test_list, fingerprint_order)
    overlapping_plot(basedir, results, test_list, fingerprint_order)

    for test in test_list:
        plotting(basedir, results, test, fingerprint_order)
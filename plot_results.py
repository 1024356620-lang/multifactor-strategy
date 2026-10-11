# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import os

os.makedirs('results', exist_ok=True)


def plot_nav():
    nav_gross = pd.read_csv('results/nav_gross.csv')['nav_gross']
    nav_net = pd.read_csv('results/nav_net.csv')['nav_net']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(nav_gross, label='毛收益净值', linewidth=2, color='#2E86AB')
    ax.plot(nav_net, label='净收益净值（扣双边千2.5）', linewidth=2, color='#E63946')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title('多因子策略净值曲线（中证500，季度调仓）', fontsize=13)
    ax.set_xlabel('调仓期数')
    ax.set_ylabel('净值')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/nav_curve.png', dpi=150)
    plt.close()
    print("已保存 results/nav_curve.png")


def plot_layered():
    layer_df = pd.read_csv('results/layered_returns.csv').set_index('date')
    layer_mean = layer_df.mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#A8DADC', '#457B9D', '#1D3557', '#457B9D', '#A8DADC']
    bars = ax.bar(layer_mean.index, layer_mean.values * 100, color=colors)
    ax.set_title('分层回测：G1（打分最低）~ G5（打分最高）', fontsize=13)
    ax.set_xlabel('分组')
    ax.set_ylabel('平均季度收益（%）')
    ax.axhline(y=0, color='gray', linewidth=0.8)
    for bar, v in zip(bars, layer_mean.values * 100):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                f'{v:.2f}%', ha='center', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/layered_bar.png', dpi=150)
    plt.close()
    print("已保存 results/layered_bar.png")


if __name__ == '__main__':
    plot_nav()
    plot_layered()
import seaborn as sns
import matplotlib.pyplot as plt

def top_show(inp_dt, col, targ_col):
    bot_10_df = inp_dt.sort_values(by=col).head(10)
    top_10_df = inp_dt.sort_values(by=col, ascending=False).head(10)
    dfs = [bot_10_df, top_10_df]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 3))
    for idx, fea in enumerate(['lowest', 'highest']):
        sns.barplot(dfs[idx], x=targ_col, y=col, ax=ax[idx])
        ax[idx].set_title(f"Top 10 {targ_col} has  the {fea} {col} cases")
        xval = 0
        for xtick, yval in zip(ax[idx].get_xticklabels(), dfs[idx].set_index(targ_col)[col]):
            try:
                xtick.set_rotation(45) 
            except:
                print(col) 
            ax[idx].text(xval, yval, str(yval), ha='center', va='bottom')
            xval += 1
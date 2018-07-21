
def str2datetime(x):
    try:
        return datetime.strptime(x, '%Y-%m-%d')
    except:
        return pd.NaT


def date_lag(var_list, bench):
    # asset var_list is list
    for item_date in var_list:
        name = item_date + '_' + bench
        df[name] = ((df[item_date].apply(str2datetime)-df[bench].apply(str2datetime))/timedelta(days=31)).apply(np.round)
        pd.pivot_table(df,index=['churn'], values=[name], aggfunc=[np.mean, max, min, np.std]).stack(level=0).reset_index(level=1).reset_index().boxplot(by='churn', title=name)
        plt.savefig('../figures/'+name)
from io import StringIO
from subprocess import check_call

from metaflow import FlowSpec, step, IncludeFile, conda_base

@conda_base(python='3.8.10',
            libraries={'scikit-learn': '1.0.2',
                       'pandas': '1.2.5'})
class TreeFeatures(FlowSpec):

    input_csv = IncludeFile('input_csv', default='emp_attrition.csv')

    @step
    def start(self):
        import pandas as pd
        self.prepare_data(pd.read_csv(StringIO(self.input_csv)))
        self.next(self.end)

    def prepare_data(self, df):
        import pandas as pd
        from sklearn.model_selection import train_test_split

        raw_target = df['Attrition']
        raw = df.drop(['Attrition', 'YearsAtCompany'], axis=1)
        categorical = []
        for col, val in raw.iteritems():
            if val.dtype == 'object':
                categorical.append(col)
        data_cat = pd.get_dummies(raw[categorical])
        data_num = raw[raw.columns.difference(categorical)]

        data = pd.concat([data_num, data_cat], axis=1)
        target = raw_target.apply(lambda x: 1 if x == 'Yes' else 0)

        self.train_data,\
            self.test_data,\
            self.train_target,\
            self.test_target = train_test_split(data,
                                                target,
                                                train_size=0.7,
                                                random_state=0)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    TreeFeatures()
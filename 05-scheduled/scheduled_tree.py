from io import StringIO
from subprocess import check_call

from metaflow import schedule, FlowSpec, step, IncludeFile, conda_base

@schedule(daily=True)
@conda_base(python='3.8.10',
            libraries={'scikit-learn': '1.0.2',
                       'pandas': '1.2.5',
                       'graphviz': '2.48.0'})
class ScheduledTree(FlowSpec):

    input_csv = IncludeFile('input_csv', default='emp_attrition.csv')

    @step
    def start(self):
        import pandas as pd
        self.prepare_data(pd.read_csv(StringIO(self.input_csv)))
        self.next(self.train)

    def prepare_data(self, df):
        import pandas as pd
        from sklearn.model_selection import train_test_split

        raw_target = df['Attrition']
        raw = df.drop(['Attrition'], axis=1)
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
    def train(self):
        from sklearn.tree import DecisionTreeClassifier
        self.params = {
            'max_leaf_nodes': None,
            'max_depth': 4,
            'max_features' : 'sqrt',
            'random_state': 0
        }
        self.model = DecisionTreeClassifier(**self.params)
        self.model.fit(self.train_data, self.train_target)
        self.next(self.eval)

    @step
    def eval(self):
        from sklearn.metrics import accuracy_score, classification_report
        self.pred = self.model.predict(self.test_data)
        self.accuracy = accuracy_score(self.test_target, self.pred)
        print('Accuracy', self.accuracy)
        print(classification_report(self.test_target, self.pred))
        self.next(self.end)

    @step
    def end(self):
        from sklearn.tree import export_graphviz
        export_graphviz(self.model,
                        out_file='tree.dot',
                        feature_names=list(self.train_data),
                        class_names=['Yes', 'No'],
                        filled=True,
                        rounded=True)
        check_call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])
        self.viz = open('tree.png', 'rb').read()

#@schedule(cron='0/5 * * * ? *')
if __name__ == '__main__':
    ScheduledTree()

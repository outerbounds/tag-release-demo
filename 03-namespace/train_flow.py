from io import StringIO
from subprocess import check_call

from metaflow import FlowSpec, step, Flow, conda_base, get_namespace

@conda_base(python='3.8.10',
            libraries={'scikit-learn': '1.0.2',
                       'pandas': '1.2.5',
                       'graphviz': '2.48.0'})
class TreeTrain(FlowSpec):

    @step
    def start(self):
        import pandas as pd
        print('Executing in namespace', get_namespace())

        feats = Flow('TreeFeatures').latest_successful_run
        self.data_lineage = feats.pathspec
        self.train_data = feats.data.train_data
        self.test_data = feats.data.test_data
        self.train_target = feats.data.train_target
        self.test_target = feats.data.test_target
        print('Data lineage', self.data_lineage)
        self.next(self.train)

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

if __name__ == '__main__':
    TreeTrain()
from io import StringIO
from subprocess import check_call

from metaflow import FlowSpec, step, IncludeFile, conda_base

class GithubTree(FlowSpec):

    @step
    def start(self):
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    GithubTree()

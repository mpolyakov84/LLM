import math
import plotly.express as px
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, f1_score
from tqdm.notebook import tqdm
import plotly.graph_objects as go
import plotly.io as pio
import torch
import torchvision.ops as ops

pio.templates.default = 'plotly_dark'


class EvaluationDataset:
    """
    EvaluationDataset analyses prediction model base on test dataset
    """
    def __init__(self, model_func,  test_dataset, cls=False):
        self.model_func = model_func
        self.test_dataset = test_dataset
        self.dict_lst = []
        self.color = 0
        self.err_df = {}
        self.title = self.make_title(model_func)
        self.guesses = []
        self.truths = []
        self.gt_lst = []
        self.cls = cls
        self.cls_dc = {}
        self.labels = []
        self.cls_true_lst = []
        self.cls_pred_lst = []
        self.cf_matrix = [[]]

    @staticmethod
    def make_title(predictor) -> str:
        return predictor.__name__.replace("__", ".").replace("_", " ").title().replace("Gpt", "GPT")

    def intersection_area(self,box1,box2):
        dx = abs(box1[0]-box2[0])
        dy = abs(box1[1]-box2[1])
        dw = abs(box1[2]+box2[2])/2
        dh = abs(box1[3]+box2[3])/2

        if dx >= dw or dy >= dh:
            return 0
        else:
            x_cross = dw-dx
            y_cross = dh-dy
            return x_cross*y_cross

    def union_area(self,box1,box2):
        mutual_square = box1[2]*box1[3]+box2[2]*box2[3]
        intersection = self.intersection_area(box1,box2)
        union = mutual_square - intersection
        return union

    def IoU(self,box1,box2):
        intersection = self.intersection_area(box1,box2)
        union = self.union_area(box1,box2)
        return intersection/union

    def filter_marix(self,matrix):
        """
        filter matrix:  each box corresponds to one box
        :param matrix: matrix of all possible IoU
        :return:
        """

        # filter rows
        for n in range(matrix.shape[0]):
            row = matrix[n, :]
            matrix[n, :] = row * (row == row.max())

        # filter columns
        for i in range(matrix.shape[1]):
            column = matrix[:, i]
            matrix[:, i] = column * (column == column.max())

        return matrix

    def get_idx_cls_mapping(self, matrix):
        idx_cls_mapping = []
        for i, row in enumerate(matrix):
            for j, col in enumerate(row):

                if col > 0:
                    idx_cls_mapping.append((i, j))
        return idx_cls_mapping

    def compare(self):
        """
        Compares prediction model with test dataset
        and calculate errors
        :return:
        """
        for item in tqdm(self.test_dataset['test']):

            # Calculating bbox IoU for many obj


            model_result = self.model_func(item['image'])

            predict = torch.tensor(model_result['bbox'])

            targets = torch.tensor(item['objects']['bbox'])
            num_obj = len(targets)
            try:
                iou_matrix = ops.box_iou(targets, predict, fmt='cxcywh')
            except IndexError:
                mIoU = 0
            else:
                filtered_iou_matrix = self.filter_marix(iou_matrix)
                mIoU = float(filtered_iou_matrix.sum())/len(targets)

                if self.cls:

                    # get list of labels
                    if len(self.labels) == 0:
                        self.labels = model_result['labels']

                    idx_cls_mapping = self.get_idx_cls_mapping(iou_matrix)
                    cls_target = item['objects']['categories']
                    cls_predict = model_result['categories']
                    # fil cls_dc
                    for mapping in idx_cls_mapping:
                        self.cls_true_lst.append(cls_target[mapping[0]])
                        self.cls_pred_lst.append(cls_predict[mapping[1]])

                        # cls_dc implementation
                        # if cls_target[mapping[0]] == cls_predict[mapping[1]]:
                        #     # check if label in dc
                        #     if cls_target[mapping[0]] not in self.cls_dc:
                        #         self.cls_dc.update({cls_target[mapping[0]]: []})
                        #
                        #     self.cls_dc[cls_target[mapping[0]]].append(True)
                        # else:
                        #     self.cls_dc[cls_target[mapping[0]]].append(False)


            self.dict_lst.append({'IoU': mIoU, 'Number of objects': num_obj})

        self.err_df = pd.DataFrame(self.dict_lst)

        # Calculate group of IoU
        size = len(self.err_df['IoU'])
        IoU_0 = self.err_df[self.err_df['IoU']==0.0]['IoU'].count().item()*100/size
        IoU_80 = self.err_df[self.err_df['IoU'] > 0.8]['IoU'].count().item()*100/size
        tmp = self.err_df[self.err_df['IoU'] > 0.4]
        IoU_50 = tmp[tmp['IoU'] < 0.5]['IoU'].count().item()*100/size

        # calculating regression validation

        average_IoU = sum(self.err_df['IoU']) / size

        self.title = (f"{self.title} results:   "
                      f"<b>IoU:</b> {average_IoU:,.3f} , Distribution: "
                      f"{IoU_0:,.1f}% (IoU(0)), {IoU_50:,.1f}%  (IoU(mid)), {IoU_80:,.1f}% (IoU(80))")

        # calculating classification assessment
        if self.cls:
            # self.cf_matrix = confusion_matrix(self.cls_true_lst, self.cls_pred_lst, labels=self.labels)

            accuracy = accuracy_score(self.cls_true_lst, self.cls_pred_lst)
            f1 = f1_score(self.cls_true_lst, self.cls_pred_lst, average='macro')

            self.title = (f"{self.title} <br> "
                          f"<b>Accuracy</b> : {accuracy:,.3f}, <b>F1</b> : {f1:,.3f} ")



        return self.err_df

    def chart_err(self):
        """
        Show chart with distribution of IoU
        :return:
        """
        # red_dots = len(self.err_df.query('color == "red"'))
        # green_dots = len(self.err_df.query('color == "green"'))
        # hit_ratio = green_dots/(red_dots + green_dots)*100

        fig = px.scatter(
            self.err_df,
            x='IoU',
            # y='aggregate error',
            # color='color',
            # color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
            title=self.title,
            # hover_data=["width", 'height'],

        )

        fig.update_layout(showlegend=False)
        fig.update_xaxes(title_text='Distribution')
        # fig.update_yaxes(title_text='Aggregate error')
        fig.show()

    def chart(self):
        """
        Show chart of dependency IoU from number of objects
        :return:
        """


        fig1 = px.strip(
            self.err_df,
            'Number of objects',
            'IoU',
            title="Dependency IoU from number of objects",

        )

        fig1.show()

    def run(self):
        """
        Launch base methods to analyse model
        :return:
        """
        self.compare()
        self.chart()
        self.chart_err()


def evaluate_model(model_func, test_dataset, cls=False):
    tester = EvaluationDataset(model_func, test_dataset, cls)
    tester.run()

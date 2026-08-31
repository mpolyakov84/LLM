import math
import plotly.express as px
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from tqdm.notebook import tqdm
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = 'plotly_dark'


class EvaluationDataset:
    """
    EvaluationDataset analyses prediction model base on test dataset
    """
    def __init__(self, model_func,  test_dataset):
        self.model_func = model_func
        self.test_dataset = test_dataset
        self.dict_lst = []
        self.color = 0
        self.err_df = {}
        self.title = self.make_title(model_func)
        self.guesses = []
        self.truths = []
        self.gt_lst = []

    @staticmethod
    def make_title(predictor) -> str:
        return predictor.__name__.replace("__", ".").replace("_", " ").title().replace("Gpt", "GPT")

    def compare(self):
        """
        Compares prediction model with test dataset
        and calculate errors
        :return:
        """
        for item in tqdm(self.test_dataset['test']):
            model_result = self.model_func(item['image'])
            diff_center = math.sqrt((model_result['x_center'] - item['x_center'])**2 +
                                    (model_result['y_center'] - item['y_center'])**2)
            he = abs(model_result['x_center'] - item['x_center'])
            we = abs(model_result['y_center'] - item['y_center'])
            h_sum = (model_result['height'] + item['height'])/2
            w_sum = (model_result['width'] + item['width'])/2

            if he < h_sum and we < w_sum:
                color = 'green'
            else:
                color = 'red'

            h_diff = abs(model_result['height'] - item['height'])
            w_diff = abs(model_result['width'] - item['width'])
            diff_err = diff_center+h_diff+w_diff

            self.guesses.append([model_result['x_center'], model_result['y_center']])
            self.truths.append([item['x_center'], item['y_center']])

            self.gt_lst.extend([
                {'truths': item['x_center'], 'guesses': model_result['x_center'], 'criteria': 'x coordinate'},
                {'truths': item['y_center'], 'guesses': model_result['y_center'], 'criteria': 'y coordinate'},
                {'truths': item['height'], 'guesses': model_result['height'], 'criteria': 'height'},
                {'truths': item['width'], 'guesses': model_result['width'], 'criteria': 'width'}
            ]
            )

            self.dict_lst.append({'aggregate error': diff_err, 'color': color, 'width': w_sum,
                                  'height': h_sum, 'diff_center': diff_center})

        self.err_df = pd.DataFrame(self.dict_lst)

        # aggregate error
        size = len(self.err_df['aggregate error'])
        average_error = sum(self.err_df['aggregate error']) / size
        mse = mean_squared_error(self.truths, self.guesses)
        r2 = r2_score(self.truths, self.guesses) * 100
        self.title = (f"{self.title} results<br><b>Error:</b> {average_error:,.3f}   "
                      f"<b>MSE:</b> {mse:,.3f}   <b>r²:</b> {r2:.2f}%")

        return self.err_df

    def chart_err(self):
        """
        Show chart with distribution of errors
        :return:
        """
        red_dots = len(self.err_df.query('color == "red"'))
        green_dots = len(self.err_df.query('color == "green"'))
        hit_ratio = green_dots/(red_dots + green_dots)*100

        fig = px.scatter(
            self.err_df,
            x='diff_center',
            y='aggregate error',
            color='color',
            color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
            title=f"<b>Errors distribution</b><br><b>Red: {red_dots:,.3f}, Green: {green_dots:,.2f}, "
                  f"Hit ratio: {hit_ratio:.2f}% </b>",
            hover_data=["width", 'height'],

        )

        fig.update_layout(showlegend=False)
        fig.update_xaxes(title_text='Diff between centers')
        fig.update_yaxes(title_text='Aggregate error')
        fig.show()

    def chart(self):
        """
        Show chart of guesses and truths data
        to visualise model prediction results
        :return:
        """
        df = pd.DataFrame(self.gt_lst)

        fig1 = px.scatter(
            df,
            'truths',
            'guesses',
            facet_col='criteria',
            facet_col_wrap=2,
            title=self.title,
            labels={'guesses': 'Guesses', 'truths': 'Truths'},
        )

        trace_line = go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash="dash", color="deepskyblue", width=2),
            name="y = x"
        )
        trace_line.update(legendgroup="trendline", showlegend=False)

        fig1.add_trace(trace_line, row="all", col="all", exclude_empty_subplots=True)

        fig1.show()

    def run(self):
        """
        Launch base methods to analyse model
        :return:
        """
        self.compare()
        self.chart()
        self.chart_err()


def evaluate_model(model_func, test_dataset):
    tester = EvaluationDataset(model_func, test_dataset)
    tester.run()

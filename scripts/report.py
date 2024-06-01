import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Calculate width of title and position
        w = self.get_string_width(title) + 6
        self.set_x((210 - w) / 2)
        # Colors of frame, background and text
        self.set_draw_color(0, 0, 0)
        self.set_fill_color(220, 220, 250)
        self.set_text_color(0, 0, 0)
        # Thickness of frame (1 mm)
        self.set_line_width(1)
        # Title
        self.cell(w, 9, title, 1, 1, 'C', 1)
        # Line break
        self.ln(10)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Text color in gray
        self.set_text_color(128)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def chapter_title(self, model_name):
        # Arial 12
        self.set_font('Arial', '', 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 6, model_name, 0, 1, 'L', 1)
        # Line break
        self.ln(4)

    def chapter_body(self, model_evaluation):

        self.set_font('Times', '', 12)

        self.multi_cell(0, 5, f"Metric: MAE if y_true <= 30, else not computed")
        self.ln()   

        self.multi_cell(0, 5, f"Mean: {round(np.mean(model_evaluation),3)}")
        self.ln()

        self.multi_cell(0, 5, f"Std: {round(np.std(model_evaluation),3)}")
        self.ln()
        
        self.set_font('', 'I')
        
    def print_chapter(self, model_evaluation, model_name):
        self.add_page()
        self.chapter_title(model_name)
        self.chapter_body(model_evaluation)

    def add_chart(self, model1, model2, model1_name, model2_name):
        self.add_page()
        self.chapter_title(f'{model1_name} x {model2_name}')
        
        differences = np.array(model2)-np.array(model1)
        rho = 1 / 10  # 1/10 pq 10-fold?
        rope = 0.01

        posterior = bayesian_correlated_t_test(differences, rho)

        x = np.linspace(posterior.ppf(0.001), posterior.ppf(0.999), 1000)
        y = posterior.pdf(x)

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label='Posterior Distribution')
        plt.fill_between(x, y, color='lightblue', alpha=0.8)
        plt.axvline(-rope, color='orange', linestyle='--')
        plt.axvline(rope, color='orange', linestyle='--')

        plt.title('Posterior Distribution of Mean Difference of the Metric (personalized)')
        plt.xlabel('Difference of Metric (personalized)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True)

        image_path = f'./dataset/images/{model1_name}_x_{model2_name}.png'
        plt.savefig(image_path)
        plt.close()

        pdf.image(image_path, x=20, y=50, w=180)

        
        prob_better = 1 - posterior.cdf(0)

        # Probabilidade de equivalência prática (usando uma ROPE de 1%)
        prob_equiv = posterior.cdf(rope) - posterior.cdf(-rope)

        self.set_font('Times', '', 12)

        self.multi_cell(0, 5, f"Probability of practical equivalence: {prob_equiv:.4f}")
        self.ln()   

        self.multi_cell(0, 5, f"Probability of {model1_name} being better than {model2_name}: {prob_better:.4f}")
        self.ln()

def bayesian_correlated_t_test(differences, rho):
    n = len(differences)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    # t distribution parameters
    dof = n - 1
    scale = std_diff / np.sqrt(n * (1 + (n - 1) * rho))
    
    # Retorns t distribution
    return stats.t(dof, loc=mean_diff, scale=scale)   

if __name__ == "__main__":

    
    file_path = './models/xgb_pipeline_2024-05-30_15_37_22.pkl'

    with open(file_path, 'rb') as file:
        xgb = pickle.load(file)

    file_path = './models/lstm_pipeline_2024-05-31_01_32_33.pkl'

    with open(file_path, 'rb') as file:
        lstm = pickle.load(file)

    file_path = './models/exponential_pipeline_2024-06-01_15_08_09.pkl'
    with open(file_path, 'rb') as file:
        expd = pickle.load(file) 

    pdf = PDF()

    title = 'Validation Report'
    pdf.set_title('Validation Report')

    pdf.print_chapter(xgb, 'xgb_pipeline_2024-05-30_15_37_22')

    pdf.print_chapter(lstm, 'lstm_pipeline_2024-05-31_01_32_33')

    pdf.print_chapter(expd, 'exponential_pipeline_2024-06-01_15_08_09')

    pdf.add_chart(lstm, xgb, 'lstm_pipeline_2024-05-31_01_32_33', 'xgb_pipeline_2024-05-30_15_37_22')

    pdf.add_chart(expd, xgb, 'exponential_pipeline_2024-06-01_15_08_09', 'xgb_pipeline_2024-05-30_15_37_22')

    pdf.add_chart(lstm, expd, 'lstm_pipeline_2024-05-31_01_32_33', 'exponential_pipeline_2024-06-01_15_08_09')

    pdf.output('Validation_Report.pdf', 'F')

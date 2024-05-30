import numpy as np
import pandas as pd
import pickle
from fpdf import FPDF
from xgb_pipeline import XGBPipeline

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

    

if __name__ == "__main__":

    
    file_path = './models/xgb_pipeline_2024-05-30_15_37_22.pkl'

    with open(file_path, 'rb') as file:
        xgb = pickle.load(file)

    pdf = PDF()

    title = 'Validation Report'
    pdf.set_title('Validation Report')

    pdf.print_chapter(xgb, 'xgb_pipeline_2024-05-30_15_37_22')

    pdf.output('Validation_Report.pdf', 'F')
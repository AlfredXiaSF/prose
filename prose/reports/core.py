import os
from os import path
import shutil
import jinja2
from .. import viz
from pathlib import Path
from ..console_utils import error, info


TEMPLATES_FOLDER = path.abspath(path.join(path.dirname(__file__), "..", "..", "latex"))
MULTIPAGE_TEMPLATE = "report.tex"
JINJA_ENV = jinja2.Environment(
    block_start_string='\BLOCK{',
    block_end_string='}',
    variable_start_string='\VAR{',
    variable_end_string='}',
    comment_start_string='\#{',
    comment_end_string='}',
    line_statement_prefix='%%',
    line_comment_prefix='%#',
    trim_blocks=True,
    autoescape=False,
    loader=jinja2.FileSystemLoader(TEMPLATES_FOLDER)
)

class Report:
    def __init__(self, template_name, data=None, style="paper", demo=False):
        self.template_name = template_name
        self._style = style
        self.template = None
        self.dpi=150

        # to be set
        self.destination = None
        self.report_name = None
        self.figure_destination = None
        self.tex_destination = None
        if data is None:
            data = {}
        self.data = data
        self.demo = demo
        self.data["__demo"] = demo

        self.template = JINJA_ENV.get_template(self.template_name)

    def style(self):
        if self._style == "paper":
            viz.paper_style()
        elif self._style == "bokeh":
            viz.bokeh_style()

    @property
    def clean_name(self):
        return self.template_name.replace(".tex", "")

    def make_report_folder(self, destination, figures=True):
        destination = Path(destination)
        destination.mkdir(exist_ok=True)
        self.destination = destination
        self.report_name = destination.stem
        if figures:
            self.figure_destination =  destination / "figures"
            self.figure_destination.mkdir(exist_ok=True)
        self.tex_destination = destination / f"{self.report_name}.tex"
        shutil.copyfile(path.join(TEMPLATES_FOLDER, "prose-report.cls"), path.join(self.destination, "prose-report.cls"))


    def make(self):
        shutil.copyfile(path.join(TEMPLATES_FOLDER, "prose-report.cls"), path.join(self.destination, "prose-report.cls"))
        latex = self.template.render(**self.data)
        multipage_template = JINJA_ENV.get_template(MULTIPAGE_TEMPLATE)
        open(self.tex_destination, "w").write(multipage_template.render(
            latexs = [latex],
            __demo = self.demo
        ))

    def compile(self, clean=True):
        cwd = os.getcwd()
        os.chdir(self.destination)
        os.system(f"pdflatex {self.report_name}")
        os.chdir(cwd)
        if clean:
            self.remove_folder()
    
    def remove_folder(self):
        pdf_name = self.destination / f"{self.report_name}.pdf"
        shutil.copy(pdf_name, self.destination.parent)
        shutil.rmtree(self.destination)


class _Report:
    def __init__(self, reports, template_name="report.tex"):
        LatexTemplate.__init__(self, template_name)
        self.reports = reports
        if isinstance(reports, str):
            self.reports = [LatexTemplate(reports)]
        self.paths = None

    def compile(self):
        cwd = os.getcwd()
        os.chdir(self.destination)
        os.system(f"pdflatex {self.report_name}")
        os.chdir(cwd)

    def make(self, destination):
        self.make_report_folder(destination, figures=False)
        shutil.copyfile(path.join(TEMPLATES_FOLDER, "prose-report.cls"), path.join(destination, "prose-report.cls"))
        tex_destination = path.join(self.destination, f"{self.report_name}.tex")

        self.paths = []
        for report in self.reports:
            info(f"making {report.clean_name} ...")
            report.make(destination)
            self.paths.append(report.tex_destination)

        open(tex_destination, "w").write(self.template.render(paths=self.paths))



def copy_figures(folder, prefix, destination):
    figures = list(Path(folder).glob("**/*.png"))
    texts = list(Path(folder).glob("**/*.txt"))
    pdfs = list(Path(folder).glob("**/*.pdf"))
    new_folder = Path(destination)
    new_folder.mkdir(exist_ok=True)
    for fig in figures:
        if ".ipynb_checkpoints" not in str(fig):
            shutil.copy(fig, new_folder / (prefix + "_" + fig.name))
    for txt in texts:
        if ".ipynb_checkpoints" not in str(txt):
            shutil.copy(txt, new_folder / (prefix + "_" + txt.name))
    for pdf in pdfs:
        if ".ipynb_checkpoints" not in str(pdf):
            shutil.copy(pdf, new_folder / (prefix + "_" + "report.pdf"))
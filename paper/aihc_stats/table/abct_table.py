from docx import Document

from .base_table import BaseTableWriter


class ABCTTableWriter(BaseTableWriter):
    def __init__(self, logger):
        super(ABCTTableWriter, self).__init__(logger)

    def write_metric_table(self, experiment2metrics, operating_point_value):
        """Write metrics table."""
        pm_path = self.table_dir / "metrics.docx"

        # metric_names = ["AUROC", f"Sensitivity@Specificity={operating_point_value}",
        #                 f"Accuracy@Specificity={operating_point_value}",
        #                 f"Specificity@Sensitivity={operating_point_value}",
        #                 f"Accuracy@Sensitivity={operating_point_value}",
        #                 "Specificity", "Sensitivity", "Accuracy", "PPV", "NPV"]
        metric_names = ["AUROC"]

        document = Document()

        table = document.add_table(rows=len(experiment2metrics)+1, cols=len(metric_names)+1)
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = "Strategy"
        for i, metric_name in enumerate(metric_names):
            hdr_cells[i+1].text = f"{metric_name} (95% CI)"

        for i, (experiment, metrics) in enumerate(experiment2metrics.items()):
            value_cells = table.rows[i+1].cells
            value_cells[0].text = experiment
            for i, metric_name in enumerate(metric_names):
                metric_dict = metrics[metric_name.lower()]
                lower = metric_dict["lower"]
                mean = metric_dict["mean"]
                upper = metric_dict["upper"]
                value_cells[i+1].text = f"{mean:.3f} ({lower:.3f}, {upper:.3f})"

        self.logger.log(f"Writing point metrics table to {pm_path}.")

        document.save(pm_path)

    def write_tables(self, experiment2metrics, operating_point_value):
        self.write_metric_table(experiment2metrics, operating_point_value)

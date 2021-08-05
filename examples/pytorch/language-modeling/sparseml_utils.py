import collections
import inspect
import math
import os
from typing import Any, Optional

import numpy
import torch
import torch.nn.functional as F

import onnxruntime
from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.optim.optimizer import ScheduledOptimizer
from sparseml.pytorch.utils import ModuleExporter, logger

from transformers.sparse import export_model, SparseMLTrainer

from transformers.file_utils import RECIPE_NAME, WEIGHTS_NAME
from transformers.models.bert.modeling_bert import BertForQuestionAnswering


class SparseMLMaskedLanguageModelingTrainer(SparseMLTrainer):
    """
    Masked language model trainer with SparseML integration

    :param recipe: recipe for model sparsification
    :param teacher: teacher model for distillation
    :param distill_hardness: ratio of loss by teacher targets (between 0 and 1)
    :param distill_temperature: temperature for distillation
    :param args, kwargs: arguments passed into parent class
    """ 
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computing loss using teacher/student distillation
        """
        if not self.recipes or self.teacher is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        outputs = model(**inputs)
        if self.teacher is None:
            loss = outputs["loss"]
        else:
            input_device = inputs["input_ids"].device
            self.teacher = self.teacher.to(input_device)
            logits_student = outputs["logits"]
            label = inputs["labels"]
            with torch.no_grad():
                teacher_output = self.teacher(**inputs)
            logits_teacher = teacher_output["logits"]
            teacher_loss = (
                F.kl_div(
                    input=F.log_softmax(logits_student / self.distill_temperature, dim=-1),
                    target=F.softmax(logits_teacher / self.distill_temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.distill_temperature ** 2)
            )
            student_loss = self.criterion(logits_student, label)
            loss = ((1 - self.distill_hardness) * student_loss) + (self.distill_hardness * teacher_loss)
        return (loss, outputs) if return_outputs else loss


class MaskedLanguageModelingModuleExporter(ModuleExporter):
    """
    Module exporter class for Question Answering
    """

    @classmethod
    def get_output_names(self, out: Any):
        # if not isinstance(out, QuestionAnsweringModelOutput):
        #     raise ValueError("Expected QuestionAnsweringModelOutput, got {type(out)}")
        expected = ["logits"]
        if numpy.any([name for name in expected if name not in out]):
            raise ValueError("Expected output names not found in model output")
        return expected

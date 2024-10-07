# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from .backbones_w_DE import BackboneModel as BackboneModel_w_DE
from .backbones_wo_DE import BackboneModel as BackboneModel_wo_DE

__all__ = ["BackboneModel_w_DE",
           "BackboneModel_wo_DE"]

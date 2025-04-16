# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 13:48:24 2025

@author: timcr
"""
import numpy as np
NDArray = np.typing.NDArray
ArrayLike = np.typing.ArrayLike
from typing import NewType, Union, List, Tuple, Sequence, Any, Optional, IO, \
SupportsIndex, Callable, Dict, TypeVar, Iterable, Dict, Type
from numbers import Number

Scalar = NewType('Scalar', Union[np.floating, np.integer, Number])
Int = NewType('Int', Union[int, np.integer])
Scalar_List = NewType('Scalar_List', List[Scalar])
Scalar_Arr = NewType('Scalar_Arr', NDArray[Scalar])
Vector = NewType('Vector', Union[Scalar_Arr, Scalar_List])
Scalar_List_2D = NewType('Scalar_List_2D', List[Scalar_List])
Matrix = NewType('Matrix', Union[Scalar_Arr, Scalar_List_2D])

Str_List = NewType('Str_List', List[str])
Int_List = NewType('Int_List', List[Int])
Int_Arr = NewType('Int_Arr', NDArray[Int])
Int_Vector = NewType('Int_Vector', Union[Int_List, Int_Arr])

T = TypeVar('T')  # Represents the input type.
U = TypeVar('U')  # Represents the output type.

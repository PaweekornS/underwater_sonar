import inspect
import torch
import torch.nn as nn
import ast
import contextlib
import ultralytics.nn.tasks as tasks
from ultralytics.utils.ops import make_divisible
import models.modules as custom_lib

def patch_yolo_engine():
    """
    Universal Factory Patch: Dynamically registers modules and 
    handles the SonarSPDConv alias strictly within this file.
    """
    # 1. Dynamic Registration
    custom_modules = []
    for name, obj in inspect.getmembers(custom_lib, inspect.isclass):
        if obj.__module__ == custom_lib.__name__:
            setattr(tasks, name, obj)
            custom_modules.append(name)
            
            # --- THE FIX ---
            # If we find SPDConv in modules.py, also register it as SonarSPDConv
            # so the patch logic below and the YAML can find it.
            if name == "SPDConv":
                setattr(tasks, "SonarSPDConv", obj)
                custom_modules.append("SonarSPDConv (Aliased)")
    
    print(f"✓ Registered custom modules: {', '.join(custom_modules)}")

    try:
        source = inspect.getsource(tasks.parse_model)
        
        # 2. Inject BiFPN Logic
        if "elif m is BiFPN_Concat2:" not in source:
            concat_branch = "elif m is Concat:\n            c2 = sum(ch[x] for x in f)"
            bifpn_branch = concat_branch + (
                "\n        elif m is BiFPN_Concat2:\n"
                "            c2 = args[0]\n"
                "            if c2 != nc:\n"
                "                c2 = make_divisible(min(c2, max_channels) * width, 8)\n"
                "            args = [[ch[x] for x in f], c2]"
            )
            source = source.replace(concat_branch, bifpn_branch)

        # 3. Inject Custom Layer Parameter Logic
        # 3. Inject Custom Layer Parameter Logic
        if "elif m is SonarSPDConv:" not in source:
            else_branch = "else:\n            c2 = ch[f]"
            custom_logic_branch = (
                "elif m in {SPDConv, SonarSPDConv, SF}:\n"
                "            c1, c2 = ch[f], args[0]\n"
                "            if c2 != nc:\n"
                "                c2 = make_divisible(min(c2, max_channels) * width, 8)\n"
                "            args = [c1, c2, *args[1:]]\n"
                "        elif m is C3k2_DRB:\n"
                "            # Handle channel scaling and inject 'n' into the module\n"
                "            c1, c2 = ch[f], args[0]\n"
                "            if c2 != nc:\n"
                "                c2 = make_divisible(min(c2, max_channels) * width, 8)\n"
                "            args = [c1, c2, n, *args[1:]]\n"
                "            n = 1  # Force engine to skip external Sequential wrapping\n"
                "        elif m is CGAF:\n"
                "            c1 = [ch[x] for x in f]\n"
                "            c2 = args[0]\n"
                "            if c2 != nc:\n"
                "                c2 = make_divisible(min(c2, max_channels) * width, 8)\n"
                "            args = [c1, c2]\n"
                "        elif m in {CoordAtt, CBAM, EMA}:\n"
                "            c1 = ch[f]\n" 
                "            c2 = c1\n"      
                "            args = [c1, *args]\n"
                "        "
            ) + else_branch
            source = source.replace(else_branch, custom_logic_branch)

        # 4. Build execution context
        exec_globals = tasks.__dict__.copy()
        exec_globals.update({
            'torch': torch, 'nn': nn, 'inspect': inspect, 'ast': ast,
            'contextlib': contextlib, 'make_divisible': make_divisible,
        })
        
        # Ensure 'SonarSPDConv' is in the globals for the exec() call
        for name in custom_modules:
            clean_name = name.split(" ")[0] # Remove '(Aliased)' suffix
            exec_globals[clean_name] = getattr(tasks, clean_name)

        exec(source, exec_globals)
        tasks.parse_model = exec_globals['parse_model']
        print("✓ Successfully applied universal monkey-patch to YOLO engine.")

    except Exception as e:
        print(f"⚠ Critical Error: {e}")

setattr(tasks, 'Upsample', nn.Upsample)
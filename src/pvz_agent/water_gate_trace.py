"""Offline, label-probed water part gates; never feeds proposals to runtime."""
import cv2
import numpy as np
from .water_pose_perception import water_fraction


def evaluate_water_offset(detector, rgb, gray, index, row, dx, dy, head_score):
    ref=detector.water_refs[index];cfg=detector.config['water_pose']
    record=dict(template_index=index,pose=ref['pose'],row=row,
        head_xy=[ref['head_xy'][0]+dx,ref['head_xy'][1]+dy],head_similarity=float(head_score),
        head_threshold=cfg['head_threshold'],part_threshold=cfg['part_threshold'],context_threshold=cfg['context_threshold'],
        part_scores=[float(head_score)],gate='head_threshold_rejected')
    if head_score<cfg['head_threshold']:return record
    for key,rect in zip(ref['part_keys'][1:],ref['parts'][1:]):
        x1,y1,x2,y2=rect;x1+=dx;x2+=dx;y1+=dy;y2+=dy
        if not (0<=x1<x2<=800 and 0<=y1<y2<=600):
            record.update(gate='part_out_of_bounds',failed_part=key);return record
        score=float(cv2.matchTemplate(gray[y1:y2,x1:x2],detector.model[key],cv2.TM_CCOEFF_NORMED)[0,0])
        record['part_scores'].append(score)
    if min(record['part_scores'])<cfg['part_threshold']:
        record['gate']='part_threshold_rejected';return record
    record['context_similarity']=float(np.mean(record['part_scores']))
    if record['context_similarity']<cfg['context_threshold']:
        record['gate']='context_threshold_rejected';return record
    if 'water_check_xyxy' in ref:
        x1,y1,x2,y2=ref['water_check_xyxy'];x1+=dx;x2+=dx;y1+=dy;y2+=dy
        if not (0<=x1<x2<=800 and 0<=y1<y2<=600):
            record['gate']='water_check_out_of_bounds';return record
        fraction=water_fraction(rgb[y1:y2,x1:x2]);record['water_fraction']=fraction
        if fraction<cfg['water_fraction_min']:
            record['gate']='water_fraction_rejected';return record
    x,y=record['head_xy'];record['gate']='passed' if 0<=x<800 and 0<=y<600 else 'head_out_of_bounds'
    return record


def trace_water(detector,image,probes):
    if image.size!=(800,600):raise ValueError('Canonical image required')
    rgb=np.asarray(image.convert('RGB'));gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    cfg=detector.config['water_pose'];records=[];local=[]
    for index,ref in enumerate(detector.water_refs):
        template=detector.model[ref['part_keys'][0]];hx,hy,hx2,hy2=ref['parts'][0]
        for row in (3,4):
            shift=(row-ref['row'])*cfg['row_spacing_pixels'];top=max(0,hy+shift-cfg['y_tolerance']);bottom=min(600,hy2+shift+cfg['y_tolerance'])
            values=cv2.matchTemplate(gray[top:bottom,cfg['search_x_min']:800],template,cv2.TM_CCOEFF_NORMED)
            maxima=(values>=cfg['head_threshold'])&(values>=cv2.dilate(values,np.ones((7,7),np.uint8)))
            for sy,sx in zip(*np.where(maxima)):
                dx=int(sx+cfg['search_x_min']-hx);dy=int(sy+top-hy)
                records.append(evaluate_water_offset(detector,rgb,gray,index,row,dx,dy,values[sy,sx]))
            for probe in probes:
                if probe['row']!=row:continue
                px,py=probe['head_xy'];base_x=cfg['search_x_min']-hx+ref['head_xy'][0];base_y=top-hy+ref['head_xy'][1]
                x1=max(0,px-25-base_x);x2=min(values.shape[1],px+26-base_x)
                y1=max(0,py-35-base_y);y2=min(values.shape[0],py+36-base_y)
                if x1>=x2 or y1>=y2:
                    local.append(dict(probe=probe,template_index=index,row=row,gate='outside_pose_search_domain',
                        allowed_head_y=[base_y,base_y+values.shape[0]-1]));continue
                _,score,_,loc=cv2.minMaxLoc(values[y1:y2,x1:x2]);sx,sy=x1+loc[0],y1+loc[1]
                r=evaluate_water_offset(detector,rgb,gray,index,row,int(sx+cfg['search_x_min']-hx),int(sy+top-hy),score)
                r.update(probe=probe,counterfactual=True,actual_local_maximum=bool(maxima[sy,sx]));local.append(r)
    actual=detector._water_proposals(rgb)
    passed=[(r['template_index'],r['row'],r['head_xy'],round(r['context_similarity'],5)) for r in records if r['gate']=='passed']
    expected=[(r['water_template_index'],r['row'],r['head_xy'],r['context_similarity']) for r in actual]
    if passed!=expected:raise AssertionError('Diagnostic gate replay diverged from frozen detector')
    return dict(schema='pvz-water-gate-trace/1',actual_head_threshold_proposals=records,probe_summaries=local,
        accepted_proposals=actual,accepted_parity_verified=True,input_sent=False,
        limits='Actual head maxima traced; below-threshold local best uses manual probes, never adds runtime candidates.')

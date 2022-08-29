import os
import math

def write_results_to_file(bboxes, confs, times, out_path, seq, experiment_id=1):
    bbox_file = os.path.join(out_path, seq+'_%03d.txt'%experiment_id)
    conf_file = os.path.join(out_path, seq+'_%03d_confidence.value'%experiment_id)
    time_file = os.path.join(out_path, seq+'_%03d_time.value'%experiment_id)

    with open(bbox_file, 'w') as fp:
        fp.write('1\n')
        for box in bboxes[1:]:
            if not math.isnan(box[0]):
                fp.write('%d,%d,%d,%d\n'%(int(box[0]),int(box[1]), int(box[2]), int(box[3])))
            else:
                fp.write('nan,nan,nan,nan\n')

    with open(conf_file, 'w') as fp:
        for c in confs:
            fp.write('%f\n'%c)

    with open(time_file, 'w') as fp:
        for t in times:
            fp.write('%f\n'%t)

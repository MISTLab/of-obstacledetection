#!/usr/bin/env python3
import os
import sys
import glob
import absl.flags as gflags
sys.path.insert(0, '../workflow/util/')
import misc_utils as misc
from common_flags import FLAGS

outfolder = "/data/models/"

def main(argv):
	# Utility main to load flags
	try:
		FLAGS(argv)  # parse flags
	except gflags.Error:
		print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
		sys.exit(1)

	s = misc.load_settings_or_use_args(FLAGS)

	# path = os.path.join(outfolder + "models2/*")
	# for item in sorted(glob.glob(path)):
	#for x in range(0, 5, 1):
	for x in ["rgb", "flow", "flow_as_rgb", "flow_as_mag"]:
		s['name'] = "final_" + x
		s['settings_fname'] = "settings_" + s['name'] + ".json"
		s['test_dir'] = "test/rgb"
		s['epochs'] = 100
		s['img_mode']  = x
		s['model_dir'] = "models_" + x
		s['eval_dir'] = "eval"
		settings_out_filename = os.path.join(outfolder, s['settings_fname'])
		misc.write_to_file(s, settings_out_filename, beautify=True)



if __name__ == "__main__":
    main(sys.argv)
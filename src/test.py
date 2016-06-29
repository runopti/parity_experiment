import platform
import sys
import datetime
import os

info = {}
info['script_name'] = sys.argv[0].split('/')[-1]
info['python_version'] = platform.python_version()
info['sys_uname'] = platform.uname()

start_time = datetime.datetime.now()
start_utime = os.times()[0]
info['start_time'] = start_time.isoformat()


pass


end_time = datetime.datetime.now()
end_utime = os.times()[0]
info['end_time'] = end_time.isoformat()
info['elapsed_time'] = str((end_time - start_time))
info['elapsed_utime'] = str((end_utime - start_utime))

with open('info.txt', 'wb') as outfile:
    for key in info.keys():
        outfile.write("#%s=%s\n" % (key, str(info[key])))


#pickle.dump(info, outfile)

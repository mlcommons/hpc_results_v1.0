import os

if 'DEBUG_SERVER' in os.environ:
    # connect to debugger
    import pydevd_pycharm

    hostname, port = os.environ['DEBUG_SERVER'].split(':')
    pydevd_pycharm.settrace(hostname, port=int(port),
                            stdoutToServer=True, stderrToServer=True)

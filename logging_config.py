from paths import LOGS

def get_logging_config(run):
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s]: %(message)s'
            },
            'short': {
                'format': '[%(levelname)s]: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'short',
                'class': 'logging.StreamHandler',
            },
            'file': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': LOGS+run+'.log'
            },
        },
        'loggers': {
            'gan': {
                'handlers': ['default', 'file'],
                'level': 'DEBUG',
                'propagate': True,
            },
            'tensorflow': {
                'handlers': ['default', 'file'],
                'level': 'DEBUG',
                'propagate': True,
            },
        }
    }

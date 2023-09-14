import logging

def create_logger(name='', ch=True, fh=False, levelname=logging.INFO, overwrite=False):
    """
    Generate logger 
    
    Parameters
    ----------
    name
        name of logger, default "".
        
    ch
        if True, add console handler output logging to console, default True.
        
    fh
        if True, add file handler output logging to file, default False.
        
    levelname
        level of logger, default logging.INFO.
        
    overwrite
        if True, overwrite the exist handler in current logger, default False.
        
    Return
    ----------
    logger
        logger generated with desired handler, logging level and name.
        
    """

    logger = logging.getLogger(name)
    logger.setLevel(levelname)
    
    if overwrite:
        for h in logger.handlers:
            logger.removeHandler(h)
    
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')

    if ch:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if fh:
        fh = logging.FileHandler(fh, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
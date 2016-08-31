import logging
import functools
from inspect import isfunction
import time
def _logging_method( cls_name, method ):
    """
    This decorator acts on a class method and inyects a call to its logger
    (that is supposed to be stored in the "logger" instance) before and after
    the execution of the method.

    :param cls_name: String
        The name of the class that has the method
    :param method: function
        The method to be decorated
    :returns: function
        The decorated function with the logger call inyected

    NOTE: This is supposed to be used in conjunction with the metaclass LoggingClass
    """

    @functools.wraps(method)
    def wrapper(self,*args,**kwargs):
        self.logger.debug("Entering method {} of class {}".format(method.__name__,cls_name))
        start_time =  time.time()
        result = method(self,*args,**kwargs)
        end_time =  time.time()
        self.logger.debug("Exiting method {} of class {}".format(method.__name__,cls_name))
        self.logger.debug("Execution of method {} took {:0.3f} seconds".format(method.__name__,
            end_time - start_time ))
        return result
    return wrapper

def _inyect_logger( __init__ ):
    """
    This decorator acts on the magic __init__ method of a class  and inyects a
    instance of logging.getLogger into the logger attribute of the class (monkey-patching
    the class in the process).

    :param __init__: function
        The __init__ magic method to be decorated
    :returns: function
        The decorated __init__ method with the logger object inyected

    NOTE: This is supposed to be used in conjunction with the metaclass LoggingClass
    """

    @functools.wraps(__init__)
    def wrapper(self,*args,**kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        return __init__(self, *args, **kwargs)
    return wrapper

class LoggingClass(type):
    """
    This metaclass uses powerfull arcane forces to make every method of the class
    that inherits from this to log the execution and termination of itself.
    """
    def __new__(meta, name, bases, attrs):
        for attrname, attr in attrs.items():
            if isfunction(attr) :
                attrs[attrname] = _logging_method(name,attr)

        attrs['__init__'] = _inyect_logger(attrs['__init__'])

        return super().__new__(meta, name, bases, attrs)

import logging
import functools
from inspect import isfunction
import sys
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
        if getattr(sys.modules[self.__class__.__module__],'__logmodule__',False):
            self.logger.debug("Entering method {} of class {}".format(method.__name__,cls_name))
            start_time =  time.time()
            result = method(self,*args,**kwargs)
            end_time =  time.time()
            self.logger.debug("Exiting method {} of class {}".format(method.__name__,cls_name))
            self.logger.debug("Execution of method {} took {:0.3f} seconds".format(method.__name__,
                end_time - start_time ))

        else:
            result = method(self,*args,**kwargs)


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

        cls = super().__new__(meta, name, bases, attrs)

        if '__init__' in attrs:
            setattr(cls,'__init__', _inyect_logger(attrs['__init__']))
        else:
            @_inyect_logger
            def dummy__init__(self,*args,**kwargs):
                pass

            # This is a little tricky because in order to preserve the __mro__ we
            # need to call super() with the correct arguments **but** we cannot
            # use self.__class__ because that will break class inheritance with
            # a fancy eternal recursion. My solution? From the deeps of hell:
            # A cool closure of the cls variable into the __init__!

            @_inyect_logger
            def super__init__(self,*args,**kwargs):
                super(cls,self).__init__(*args,**kwargs)

            for base in bases:
                if '__init__' in base.__dict__:
                    setattr(cls,'__init__', super__init__)
                    break
            else:
                setattr(cls,'__init__',  dummy__init__)

        return cls

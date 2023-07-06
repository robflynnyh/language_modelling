from setuptools import setup

setup(
    name='lming',
    version='0.0.0.1',    
    description='Code for training language models',
    url='https://github.com/robflynnyh/language_modelling',
    author='Rob Flynn',
    author_email='rjflynn2@sheffield.ac.uk',
    license='MIT',
    packages=['lming'],
    install_requires=['torch',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)

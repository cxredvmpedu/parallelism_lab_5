project(
  'lab_5',
  'c',
  default_options: [
    'c_std=c23',
    'buildtype=debug',
    'warning_level=3',
  ],
)

src = ['src/main.c', 'src/data.c']
inc = include_directories('include')

mpi_dep = dependency('mpi', language: 'c')
dep = [mpi_dep]

executable('lab_5', sources: src, include_directories: inc, dependencies: dep)

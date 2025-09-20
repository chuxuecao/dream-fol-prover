import Lake
open Lake DSL

package «lean-fol» {
  -- add package configuration options here
}

@[default_target]
lean_lib «LeanFol» {
  -- add library configuration options here
}


require mathlib from git "https://github.com/leanprover-community/mathlib4"

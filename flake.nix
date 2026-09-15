{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            uv
            python311
            just
            ruff
            # For packages/eqty-lineage-nemo-relay, which is a Rust cdylib rather than a Python
            # package. Kept as the nixpkgs toolchain rather than rustup so CI and the dev shell
            # build with the same compiler.
            cargo
            rustc
            rustfmt
            clippy
          ];

          shellHook = ''
            export UV_PROJECT_ENVIRONMENT=".venv"
            if [ -f .venv/bin/activate ]; then
              . .venv/bin/activate
            fi
          '';
        };
      });
}

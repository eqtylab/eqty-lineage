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

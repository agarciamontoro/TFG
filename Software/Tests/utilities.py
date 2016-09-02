from numpy import sqrt,sin,cos


def override_initial_conditions(self, r, cam_theta, cam_phi, theta_cs, phi_cs):
        # Calculate initial vector direction

        Nx = sin(theta_cs) * cos(phi_cs)
        Ny = sin(theta_cs) * sin(phi_cs)
        Nz = cos(theta_cs)

        # Convert the direction of motion to the FIDO's spherical orthonormal
        # basis. See (A.10)
        nR = Nx
        nTheta = Nz
        nPhi = Ny

        # Get canonical momenta

        ro = self.kerr.ro
        delta = self.kerr.delta
        pomega = self.kerr.pomega
        alpha = self.kerr.alpha
        omega = self.kerr.omega

        # Compute energy as measured by the FIDO. See (A.11)
        E = 1

        # Compute the canonical momenta. See (A.11)
        pR = E * ro * nR / sqrt(delta)
        pTheta = E * ro * nTheta
        pPhi = E * pomega * nPhi
        # Set conserved quantities

        # Simplify notation
        theta = cam_theta
        a2 = self.blackHole.a2

        # Set conserved quantities. See (A.12)
        b = pPhi
        q = pTheta**2 + cos(theta)**2*(b**2 / sin(theta)**2 - a2)

        # HACK THE INITIAL CONDITIONS

        self.systemState[:,:,0] = r
        self.systemState[:,:,1] = cam_theta
        self.systemState[:,:,2] = cam_phi
        self.systemState[:,:,3] = pR
        self.systemState[:,:,4] = pTheta

        self.constants[:,:,0] = b
        self.constants[:,:,1] = q


from .uvp_masks import uvp_masks
import numpy as np
import bathy_smoother



def smoothing_Laplacian_rx0(MSK, Hobs, rx0max):
    """
    This program use Laplacian filter.
    The bathymetry is optimized for a given rx0 factor by doing an iterated
    sequence of Laplacian filterings.

    Usage:
    RetBathy = smoothing_Laplacian_rx0(MSK, Hobs, rx0max)

    ---MSK(eta_rho,xi_rho) is the mask of the grid
         1 for sea
         0 for land
    ---Hobs(eta_rho,xi_rho) is the raw depth of the grid
    ---rx0max is the target rx0 roughness factor
    """

    eta_rho, xi_rho = Hobs.shape

    ListNeigh = np.array([[1, 0],
                          [0, 1],
                          [-1, 0],
                          [0, -1]])

    RetBathy = Hobs.copy()

    tol = 0.00001
    WeightMatrix = np.zeros((eta_rho, xi_rho))
    for iEta in range(eta_rho):
        for iXi in range(xi_rho):
            WeightSum = 0
            for ineigh in range(4):
                iEtaN = iEta + ListNeigh[ineigh,0]
                iXiN = iXi + ListNeigh[ineigh,1]
                if (iEtaN <= eta_rho-1 and iEtaN >= 0 and iXiN <= xi_rho-1 \
                      and iXiN >= 0 and MSK[iEtaN,iXiN] == 1):
                    WeightSum = WeightSum + 1

            WeightMatrix[iEta,iXi] = WeightSum

    Iter = 1
    NumberDones = np.zeros((eta_rho, xi_rho))
    while(True):
        RoughMat = bathy_smoother.bathy_tools.RoughnessMatrix(RetBathy, MSK)
        Kbefore = np.where(RoughMat > rx0max)
        nbPtBefore = np.size(Kbefore, 1)
        realR = RoughMat.max()
        TheCorrect = np.zeros((eta_rho,xi_rho))
        IsFinished = 1
        nbPointMod = 0
        AdditionalDone = np.zeros((eta_rho, xi_rho))
        for iEta in range(eta_rho):
            for iXi in range(xi_rho):
                Weight = 0
                WeightSum = 0
                for ineigh in range(4):
                    iEtaN = iEta + ListNeigh[ineigh,0]
                    iXiN = iXi + ListNeigh[ineigh,1]
                    if (iEtaN <= eta_rho-1 and iEtaN >= 0 and iXiN <= xi_rho-1 \
                          and iXiN >= 0 and MSK[iEtaN,iXiN] == 1):
                        Weight = Weight + RetBathy[iEtaN,iXiN]
                        AdditionalDone[iEtaN,iXiN] = AdditionalDone[iEtaN,iXiN] + NumberDones[iEta,iXi]

                TheWeight = WeightMatrix[iEta,iXi]
                WeDo = 0
                if TheWeight > tol:
                    if RoughMat[iEta,iXi] > rx0max:
                        WeDo = 1
                    if NumberDones[iEta,iXi] > 0:
                        WeDo = 1

                if WeDo == 1:
                    IsFinished = 0
                    TheDelta = (Weight - TheWeight * RetBathy[iEta,iXi]) / (2 * TheWeight)
                    TheCorrect[iEta,iXi] = TheCorrect[iEta,iXi] + TheDelta
                    nbPointMod = nbPointMod + 1
                    NumberDones[iEta,iXi] = 1

        NumberDones = NumberDones + AdditionalDone
        RetBathy = RetBathy + TheCorrect
        NewRoughMat = bathy_smoother.bathy_tools.RoughnessMatrix(RetBathy, MSK)
        Kafter = np.where(NewRoughMat > rx0max)
        nbPtAfter = np.size(Kafter, 1)
        TheProd = (RoughMat > rx0max) * (NewRoughMat > rx0max)
        nbPtInt = TheProd.sum()
        if (nbPtInt == nbPtAfter and nbPtBefore == nbPtAfter):
            eStr=' no erase'
        else:
            eStr='';
            NumberDones = np.zeros((eta_rho, xi_rho))

        print('Iteration #', Iter)
        print('current r=', realR, '  nbPointMod=', nbPointMod, eStr)
        print(' ')

        Iter = Iter + 1

        if (IsFinished == 1):
            break

    return RetBathy
import numpy as np
import math
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def l2(adv_patch, orig_patch):
    assert adv_patch.shape == orig_patch.shape
    return np.sum((adv_patch - orig_patch) ** 2)


def linf(adv_patch, orig_patch):
    assert adv_patch.shape == orig_patch.shape
    return float(np.max(np.abs(adv_patch - orig_patch)))


def project_linf(adv_patch, orig_patch, eps_linf):
    assert adv_patch.shape == orig_patch.shape
    eps_linf = float(eps_linf)
    if eps_linf > 0 and np.issubdtype(orig_patch.dtype, np.floating):
        dtype_eps = np.array(eps_linf, dtype=orig_patch.dtype)
        zero = np.array(0., dtype=orig_patch.dtype)
        eps_linf = float(np.nextafter(dtype_eps, zero, dtype=orig_patch.dtype))

    lo = np.maximum(orig_patch - eps_linf, 0.)
    hi = np.minimum(orig_patch + eps_linf, 1.)
    return np.clip(adv_patch, lo, hi).astype(orig_patch.dtype, copy=False)


def sh_selection(n_queries, it):
    """ schedule to decrease the parameter p """

    t = max((float(n_queries - it) / n_queries - .0) ** 1., 0) * .75

    return t


def update_location(loc_new, h_i, h, s):
    loc_new += np.random.randint(low=-h_i, high=h_i + 1, size=(2,))
    loc_new = np.clip(loc_new, 0, h - s)
    return loc_new


def render(x, w):
    phenotype = np.ones((w, w, 3))
    radius_avg = (phenotype.shape[0] + phenotype.shape[1]) / 2 / 6
    for row in x:
        overlay = phenotype.copy()
        cv2.circle(
            overlay,
            center=(int(row[1] * w), int(row[0] * w)),
            radius=int(row[2] * radius_avg),
            color=(int(row[3] * 255), int(row[4] * 255), int(row[5] * 255)),
            thickness=-1,
        )
        alpha = row[6]
        phenotype = cv2.addWeighted(overlay, alpha, phenotype, 1 - alpha, 0)

    return phenotype / 255.


def render_many(genotypes, w):
    return np.stack([render(genotype, w) for genotype in genotypes], axis=0)


def apply_patch(x, patch, loc):
    s = patch.shape[0]
    x_adv = x.copy()
    x_adv[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :] = patch
    np.clip(x_adv, 0., 1., out=x_adv)
    return x_adv


def apply_patch_batch(x, patches, locs):
    x_batch = np.repeat(x[None, ...], len(locs), axis=0)
    for idx, (patch, loc) in enumerate(zip(patches, locs)):
        s = patch.shape[0]
        x_batch[idx, loc[0]: loc[0] + s, loc[1]: loc[1] + s, :] = patch
    np.clip(x_batch, 0., 1., out=x_batch)
    return x_batch


def save_rgb_image(path, image):
    image_uint8 = (np.clip(image, 0., 1.) * 255).round().astype(np.uint8)
    Image.fromarray(image_uint8).save(path)


def mutate(soln, mut):
    """Mutates specie for evolution.

    Args:
        specie (species.Specie): Specie to mutate.

    Returns:
        New Specie class, that has been mutated.
        :param soln:
    """
    new_specie = soln.copy()

    # Randomization for Evolution
    genes = soln.shape[0]
    length = soln.shape[1]
    y = np.random.randint(0, genes)
    change = np.random.randint(0, length + 1)

    if change >= length + 1:
        change -= 1
        i, j = y, np.random.randint(0, genes)
        i, j, s = (i, j, -1) if i < j else (j, i, 1)
        new_specie[i: j + 1] = np.roll(new_specie[i: j + 1], shift=s, axis=0)
        y = j

    selection = np.random.choice(length, size=change, replace=False)

    if np.random.rand() < mut:
        new_specie[y, selection] = np.random.rand(len(selection))
    else:
        new_specie[y, selection] += (np.random.rand(len(selection)) - 0.5) / 3
        new_specie[y, selection] = np.clip(new_specie[y, selection], 0, 1)

    return new_specie


class Attack:
    def __init__(self, params):
        self.params = params
        self.process = []
        self.eval_batch_size = max(1, int(params.get("eval_batch_size", 1)))
        self.parallel_locations = bool(params.get("parallel_locations", False))
        self.fixed_location = bool(params.get("fixed_location", False))
        self.initial_loc = params.get("initial_loc")
        self.trace_every = int(params.get("trace_every", 1))
        self.eps_linf = params.get("eps_linf")
        if self.eps_linf is not None:
            self.eps_linf = float(self.eps_linf)
            if self.eps_linf < 0:
                raise ValueError("eps_linf must be non-negative")

    def _initial_location(self, h, s):
        if self.initial_loc is not None:
            loc = np.asarray(self.initial_loc, dtype=np.int64)
            if loc.shape != (2,):
                raise ValueError(f"initial_loc must have shape (2,), got {loc.shape}")
            return np.clip(loc, 0, h - s)

        if h <= s:
            return np.zeros(2, dtype=np.int64)
        return np.random.randint(h - s, size=2)

    def _record_process(self, query, loc, patch_geno):
        if self.trace_every > 0 and query % self.trace_every == 0:
            self.process.append([loc.copy(), patch_geno.copy()])

    def _orig_patch(self, x, loc, s):
        return x[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :]

    def _project_patch(self, x, patch, loc):
        if self.eps_linf is None:
            return patch
        s = patch.shape[0]
        return project_linf(patch, self._orig_patch(x, loc, s), self.eps_linf)

    def _render_patch(self, patch_geno, s, x, loc):
        return self._project_patch(x, render(patch_geno, s), loc)

    def _render_patch_many(self, genotypes, s, x, locs):
        patches = render_many(genotypes, s)
        if self.eps_linf is None:
            return patches
        return np.stack([
            project_linf(patch, self._orig_patch(x, loc, s), self.eps_linf)
            for patch, loc in zip(patches, locs)
        ], axis=0)

    def completion_procedure(self, adversarial, x_adv, queries, loc, patch, loss_function):
        s = patch.shape[0]
        orig_patch = self._orig_patch(self.params["x"], loc, s)
        data = {
            "orig": self.params["x"],
            "adversary": x_adv,
            "adversarial": adversarial,
            "queries": queries,
            "loc": loc,
            "patch": patch,
            "patch_width": int(math.ceil(self.params["eps"] ** .5)),
            "eps_linf": self.eps_linf,
            "attack_model": self.params.get("attack_model"),
            "attack_model_source": self.params.get("attack_model_source"),
            "attack_model_index": self.params.get("attack_model_index"),
            "fixed_location": self.fixed_location,
            "location_source": self.params.get("location_source"),
            "initial_loc": None if self.initial_loc is None else np.asarray(self.initial_loc).copy(),
            "final_l2": l2(patch, orig_patch),
            "final_linf": linf(patch, orig_patch),
            "final_prediction": loss_function.get_label(x_adv),
            "process": self.process
        }

        save_path = Path(self.params["save_directory"])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, data, allow_pickle=True)

        if self.params.get("save_images", True):
            image_stem = save_path.with_suffix("")
            save_rgb_image(image_stem.with_name(image_stem.name + "_adversary.png"), x_adv)
            save_rgb_image(image_stem.with_name(image_stem.name + "_patch.png"), patch)

    def _optimise_sequential(self, loss_function):
        # This branch intentionally mirrors the original CamoPatch loop.
        x = self.params["x"]
        c, h, w = self.params["c"], self.params["h"], self.params["w"]
        eps = self.params["eps"]
        s = int(math.ceil(eps ** .5))

        patch_geno = np.random.rand(self.params["N"], 7)
        loc = self._initial_location(h, s)
        patch = self._render_patch(patch_geno, s, x, loc)

        update_loc_period = self.params["update_loc_period"]

        x_adv = apply_patch(x, patch, loc)
        adversarial, loss = loss_function(x_adv)
        l2_curr = l2(adv_patch=patch, orig_patch=self._orig_patch(x, loc, s).copy())

        patch_counter = 0
        n_queries = self.params["n_queries"]
        for it in tqdm(range(1, n_queries)):
            patch_counter += 1
            if self.fixed_location or patch_counter < update_loc_period:
                patch_new_geno = mutate(patch_geno, self.params["mut"])
                patch_new = self._render_patch(patch_new_geno, s, x, loc)
                x_adv_new = apply_patch(x, patch_new, loc)

                adversarial_new, loss_new = loss_function(x_adv_new)

                orig_patch = self._orig_patch(x, loc, s).copy()
                l2_new = l2(adv_patch=patch_new, orig_patch=orig_patch)

                if adversarial == True and adversarial_new == True:

                    if l2_new < l2_curr:
                        loss = loss_new
                        adversarial = adversarial_new
                        patch = patch_new
                        patch_geno = patch_new_geno
                        x_adv = x_adv_new
                        l2_curr = l2_new

                else:
                    if loss_new < loss:  # minimization
                        loss = loss_new
                        adversarial = adversarial_new
                        patch = patch_new
                        patch_geno = patch_new_geno
                        x_adv = x_adv_new
                        l2_curr = l2_new

            else:
                patch_counter = 0

                sh_i = int(max(sh_selection(n_queries, it) * h, 0))
                loc_new = loc.copy()
                loc_new = update_location(loc_new, sh_i, h, s)
                patch_new = self._render_patch(patch_geno, s, x, loc_new)
                x_adv_new = apply_patch(x, patch_new, loc_new)

                adversarial_new, loss_new = loss_function(x_adv_new)

                orig_patch_new = self._orig_patch(x, loc_new, s).copy()
                l2_new = l2(adv_patch=patch_new, orig_patch=orig_patch_new)

                if adversarial == True and adversarial_new == True:
                    if l2_new < l2_curr:
                        loss = loss_new
                        adversarial = adversarial_new
                        loc = loc_new
                        patch = patch_new

                        x_adv = x_adv_new
                        l2_curr = l2_new

                else:
                    diff = loss_new - loss
                    curr_temp = self.params["temp"] / (it + 1)
                    metropolis = math.exp(-diff / curr_temp)

                    if loss_new < loss or np.random.rand() < metropolis:
                        loss = loss_new
                        adversarial = adversarial_new
                        loc = loc_new
                        patch = patch_new
                        x_adv = x_adv_new
                        l2_curr = l2_new

            if self.trace_every > 0 and it % self.trace_every == 0:
                self.process.append([loc, patch_geno])

        self.completion_procedure(adversarial, x_adv, it, loc, patch, loss_function)
        return

    def _accept_patch_update(self, adversarial, loss, l2_curr, adversarial_new, loss_new, l2_new):
        if adversarial and adversarial_new:
            return l2_new < l2_curr
        return loss_new < loss

    def _accept_location_update(self, adversarial, loss, l2_curr, adversarial_new, loss_new, l2_new, query):
        if adversarial and adversarial_new:
            return l2_new < l2_curr

        if loss_new < loss:
            return True

        curr_temp = self.params["temp"] / (query + 1)
        if curr_temp <= 0:
            return False
        metropolis = math.exp(-(loss_new - loss) / curr_temp)
        return np.random.rand() < metropolis

    def _patch_update(self, x, s, loc, patch_geno, patch, adversarial, loss, l2_curr, loss_function):
        patch_new_geno = mutate(patch_geno, self.params["mut"])
        patch_new = self._render_patch(patch_new_geno, s, x, loc)
        x_adv_new = apply_patch(x, patch_new, loc)
        adversarial_new, loss_new = loss_function(x_adv_new)
        orig_patch = self._orig_patch(x, loc, s)
        l2_new = l2(adv_patch=patch_new, orig_patch=orig_patch)

        if self._accept_patch_update(adversarial, loss, l2_curr, adversarial_new, loss_new, l2_new):
            return patch_new_geno, patch_new, x_adv_new, adversarial_new, loss_new, l2_new
        return patch_geno, patch, None, adversarial, loss, l2_curr

    def _location_update(self, x, h, s, loc, patch_geno, patch, adversarial, loss, l2_curr, query, loss_function):
        sh_i = int(max(sh_selection(self.params["n_queries"], query) * h, 0))
        loc_new = update_location(loc.copy(), sh_i, h, s)
        patch_new = self._render_patch(patch_geno, s, x, loc_new)
        x_adv_new = apply_patch(x, patch_new, loc_new)
        adversarial_new, loss_new = loss_function(x_adv_new)
        orig_patch_new = self._orig_patch(x, loc_new, s)
        l2_new = l2(adv_patch=patch_new, orig_patch=orig_patch_new)

        if self._accept_location_update(adversarial, loss, l2_curr, adversarial_new, loss_new, l2_new, query):
            return loc_new, patch_new, x_adv_new, adversarial_new, loss_new, l2_new
        return loc, patch, None, adversarial, loss, l2_curr

    def _patch_update_batch(self, x, s, loc, patch_geno, patch, adversarial, loss, l2_curr, batch_size, loss_function):
        genotypes = [mutate(patch_geno, self.params["mut"]) for _ in range(batch_size)]
        locs = np.repeat(loc[None, :], batch_size, axis=0)
        patches = self._render_patch_many(genotypes, s, x, locs)
        x_batch = apply_patch_batch(x, patches, locs)
        adversarials_new, losses_new = loss_function.batch(x_batch)
        orig_patch = self._orig_patch(x, loc, s)
        l2s_new = np.array([l2(adv_patch=patch_new, orig_patch=orig_patch) for patch_new in patches])

        if adversarial:
            candidates = [
                idx for idx, (adv_new, l2_new) in enumerate(zip(adversarials_new, l2s_new))
                if adv_new and l2_new < l2_curr
            ]
            if candidates:
                best_idx = min(candidates, key=lambda idx: l2s_new[idx])
                return (
                    genotypes[best_idx],
                    patches[best_idx],
                    x_batch[best_idx],
                    adversarials_new[best_idx],
                    losses_new[best_idx],
                    float(l2s_new[best_idx]),
                )
            candidates = [
                idx for idx, (adv_new, loss_new) in enumerate(zip(adversarials_new, losses_new))
                if not adv_new and loss_new < loss
            ]
            if candidates:
                best_idx = min(candidates, key=lambda idx: losses_new[idx])
                return (
                    genotypes[best_idx],
                    patches[best_idx],
                    x_batch[best_idx],
                    adversarials_new[best_idx],
                    losses_new[best_idx],
                    float(l2s_new[best_idx]),
                )
        else:
            losses_arr = np.asarray(losses_new)
            best_idx = int(np.argmin(losses_arr))
            if losses_arr[best_idx] < loss:
                return (
                    genotypes[best_idx],
                    patches[best_idx],
                    x_batch[best_idx],
                    adversarials_new[best_idx],
                    float(losses_arr[best_idx]),
                    float(l2s_new[best_idx]),
                )

        return patch_geno, patch, None, adversarial, loss, l2_curr

    def _location_update_batch(self, x, h, s, loc, patch_geno, patch, adversarial, loss, l2_curr, query, batch_size, loss_function):
        sh_i = int(max(sh_selection(self.params["n_queries"], query) * h, 0))
        locs = np.stack([update_location(loc.copy(), sh_i, h, s) for _ in range(batch_size)], axis=0)
        if self.eps_linf is None:
            patches = np.repeat(patch[None, ...], batch_size, axis=0)
        else:
            genotypes = [patch_geno for _ in range(batch_size)]
            patches = self._render_patch_many(genotypes, s, x, locs)
        x_batch = apply_patch_batch(x, patches, locs)
        adversarials_new, losses_new = loss_function.batch(x_batch)
        l2s_new = np.array([
            l2(adv_patch=patch_new, orig_patch=self._orig_patch(x, loc_new, s))
            for patch_new, loc_new in zip(patches, locs)
        ])

        if adversarial:
            candidates = [
                idx for idx, (adv_new, l2_new) in enumerate(zip(adversarials_new, l2s_new))
                if adv_new and l2_new < l2_curr
            ]
            if candidates:
                best_idx = min(candidates, key=lambda idx: l2s_new[idx])
                return (
                    locs[best_idx],
                    patches[best_idx],
                    x_batch[best_idx],
                    adversarials_new[best_idx],
                    losses_new[best_idx],
                    float(l2s_new[best_idx]),
                )
            curr_temp = self.params["temp"] / (query + 1)
            accepted = []
            for idx, (adv_new, loss_new) in enumerate(zip(adversarials_new, losses_new)):
                if adv_new:
                    continue
                if loss_new < loss:
                    accepted.append(idx)
                elif curr_temp > 0:
                    metropolis = math.exp(-(loss_new - loss) / curr_temp)
                    if np.random.rand() < metropolis:
                        accepted.append(idx)

            if accepted:
                best_idx = min(accepted, key=lambda idx: losses_new[idx])
                return (
                    locs[best_idx],
                    patches[best_idx],
                    x_batch[best_idx],
                    adversarials_new[best_idx],
                    losses_new[best_idx],
                    float(l2s_new[best_idx]),
                )
        else:
            curr_temp = self.params["temp"] / (query + 1)
            accepted = []
            for idx, loss_new in enumerate(losses_new):
                if loss_new < loss:
                    accepted.append(idx)
                elif curr_temp > 0:
                    metropolis = math.exp(-(loss_new - loss) / curr_temp)
                    if np.random.rand() < metropolis:
                        accepted.append(idx)

            if accepted:
                best_idx = min(accepted, key=lambda idx: losses_new[idx])
                return (
                    locs[best_idx],
                    patches[best_idx],
                    x_batch[best_idx],
                    adversarials_new[best_idx],
                    losses_new[best_idx],
                    float(l2s_new[best_idx]),
                )

        return loc, patch, None, adversarial, loss, l2_curr

    def optimise(self, loss_function):
        if self.eval_batch_size == 1 and not self.parallel_locations:
            return self._optimise_sequential(loss_function)

        # initialize
        x = self.params["x"]
        h, w = self.params["h"], self.params["w"]
        eps = self.params["eps"]
        s = int(math.ceil(eps ** .5))

        patch_geno = np.random.rand(self.params["N"], 7)
        loc = self._initial_location(h, s)
        patch = self._render_patch(patch_geno, s, x, loc)

        update_loc_period = self.params["update_loc_period"]

        x_adv = apply_patch(x, patch, loc)
        adversarial, loss = loss_function(x_adv)
        l2_curr = l2(adv_patch=patch, orig_patch=self._orig_patch(x, loc, s))

        patch_counter = 0
        n_queries = self.params["n_queries"]
        query = 1
        pbar = tqdm(total=n_queries - 1)

        while query < n_queries:
            patch_counter += 1

            if self.fixed_location or patch_counter < update_loc_period:
                if self.fixed_location:
                    batch_size = min(self.eval_batch_size, n_queries - query)
                else:
                    batch_size = min(self.eval_batch_size, update_loc_period - patch_counter, n_queries - query)
                if batch_size == 1:
                    patch_geno, patch, x_adv_new, adversarial, loss, l2_curr = self._patch_update(
                        x, s, loc, patch_geno, patch, adversarial, loss, l2_curr, loss_function
                    )
                else:
                    patch_geno, patch, x_adv_new, adversarial, loss, l2_curr = self._patch_update_batch(
                        x, s, loc, patch_geno, patch, adversarial, loss, l2_curr, batch_size, loss_function
                    )
                if not self.fixed_location:
                    patch_counter += batch_size - 1

            else:
                patch_counter = 0
                batch_size = min(self.eval_batch_size, n_queries - query) if self.parallel_locations else 1
                if batch_size == 1:
                    loc, patch, x_adv_new, adversarial, loss, l2_curr = self._location_update(
                        x, h, s, loc, patch_geno, patch, adversarial, loss, l2_curr, query, loss_function
                    )
                else:
                    loc, patch, x_adv_new, adversarial, loss, l2_curr = self._location_update_batch(
                        x, h, s, loc, patch_geno, patch, adversarial, loss, l2_curr, query, batch_size, loss_function
                    )

            if x_adv_new is not None:
                x_adv = x_adv_new
            query += batch_size
            self._record_process(query, loc, patch_geno)
            pbar.update(batch_size)

        pbar.close()
        self.completion_procedure(adversarial, x_adv, query, loc, patch, loss_function)
        return

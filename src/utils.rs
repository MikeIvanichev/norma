pub struct InclusiveBoxedBy<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    v: &'a [T],
    pred: P,
    finished: bool,
}

impl<'a, T: 'a, P: FnMut(&T) -> bool> InclusiveBoxedBy<'a, T, P> {
    #[inline]
    pub(super) fn new(slice: &'a [T], pred: P) -> Self {
        let finished = slice.is_empty();
        Self {
            v: slice,
            pred,
            finished,
        }
    }
}

impl<'a, T, P> Iterator for InclusiveBoxedBy<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.finished {
            return None;
        }

        if let Some(s_idx) = self.v.iter().position(|x| (self.pred)(x)) {
            if let Some(e_idx) = self.v[s_idx + 1..]
                .iter()
                .position(|x| (self.pred)(x))
                .map(|idx| s_idx + idx + 2)
            {
                let ret = Some(&self.v[s_idx..e_idx]);
                self.v = &self.v[e_idx..];
                return ret;
            }
        }

        self.finished = true;
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            // If the predicate doesn't match anything, we yield one slice.
            // If it matches every element, we yield `len()` one-element slices,
            // or a single empty slice.
            (1, Some(self.v.len() / 2))
        }
    }
}

pub trait SliceExt<T> {
    fn inclusive_boxed_by<F>(&self, pred: F) -> InclusiveBoxedBy<'_, T, F>
    where
        F: FnMut(&T) -> bool;
}

impl<T> SliceExt<T> for [T] {
    fn inclusive_boxed_by<F>(&self, pred: F) -> InclusiveBoxedBy<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        InclusiveBoxedBy::new(self, pred)
    }
}
